from mcp.server.lowlevel import NotificationOptions, Server
import os
import logging
import dotenv
import asyncio
from langchain_community.document_loaders import PyPDFLoader
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    GetPromptResult,
    Prompt,
    PromptMessage,
    PromptArgument,
    ServerCapabilities,
    ResourcesCapability,
    ToolsCapability,
    ClientNotification,
    ToolListChangedNotification,
    #ToolsChangedParams,
    ResourceListChangedNotification,
    ResourceUpdatedNotificationParams,
    ResourceUpdatedNotification,
    ServerNotification,
    PromptListChangedNotification,
    #ResourcesChangedParams,
    LoggingLevel,
    LoggingCapability,
)
from mcp.server.models import InitializationOptions
from core.logger import logger
from pydantic.error_wrappers import ValidationError
from core.schemas import CHARGE_POINT_LOCATOR_SCHEMA, EV_TRIP_PLANNER_SCHEMA, FindChargingStationsPrompt, ChargingTimeEstimatePrompt, RoutePlannerPrompt
from agentTools.charge_station_locator import charge_points_locator
from agentTools.ev_trip_planner import ev_trip_planner
import glob
import time
from collections import defaultdict
dotenv.load_dotenv()
from mcp.server.session import ServerSession
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import uvicorn
import mcp.server.stdio
server = Server("MCP EV Assistant Server")


# Mapping of URIs to sets of client IDs
resource_subscribers: dict[str, set[ServerSession]] = defaultdict(set)
client_sessions: dict[str, ServerSession] = {}

async def store_client_session_if_needed():
    """
    Store the client session if it hasn't been stored yet.
    """
    client_session = server.request_context.session
    if client_session not in client_sessions:
        client_sessions[client_session] = client_session
        logger.info(f"Stored session for client: {client_session}")
    else:
        logger.info(f"Session already stored for client: {client_session}")
    logger.info(f"Client sessions: {client_sessions}")

@server.set_logging_level()
async def handle_set_logging_level(level: LoggingLevel) -> None:
    """Handle logging level changes"""
    try:
        # Convert MCP logging level to Python logging level
        python_level = {
            LoggingLevel.DEBUG: logging.DEBUG,
            LoggingLevel.INFO: logging.INFO,
            LoggingLevel.WARNING: logging.WARNING,
            LoggingLevel.ERROR: logging.ERROR,
        }.get(level, logging.INFO)
        
        # Set the logging level
        logger.setLevel(python_level)
        logger.info(f"Logging level set to {level}")
    except Exception as e:
        logger.error(f"Error setting logging level: {e}")



def add_subscriber(uri: str) -> None:
    """Add a client to the list of subscribers for a URI."""
    client_session = server.request_context.session
    resource_subscribers[uri].add(client_session)

def remove_subscriber(uri: str) -> None:
    """Remove a client from the list of subscribers for a URI."""
    client_session = server.request_context.session
    if uri in resource_subscribers:
        resource_subscribers[uri].discard(client_session)
        if not resource_subscribers[uri]:  # Remove the entry if no subscribers remain
            del resource_subscribers[uri]
            logger.info(f"Removed resource {uri} from resource_subscribers")





@server.subscribe_resource()
async def handle_subscribe_resource(uri: str) -> None:
    logger.info(f"Client {server.request_context.session} subscribed to resource: {uri}")
    add_subscriber(uri)

@server.unsubscribe_resource()
async def handle_unsubscribe_resource(uri: str) -> None:
    logger.info(f"handle_unsubscribe_resource: {uri}")  
    remove_subscriber(uri)


async def notify_resource_change(uri: str):
    """Notify subscribers about resource changes"""
    try:
        if uri in resource_subscribers:
            # Create a notification message
            notification = ServerNotification(
                ResourceUpdatedNotification(
                    method="notifications/resources/updated",
                    params=ResourceUpdatedNotificationParams(uri=uri),
                )
            )
            
            # Send notification to all subscribers of this resource
            for client_session in resource_subscribers[uri]:
                try:
                    if client_session:
                        await client_session.send_notification(notification)
                        logger.info(f"Sent resource change notification to client {client_session}")
                    else:
                        logger.error("Client not found in client_sessions")
                except Exception as e:
                    logger.error(f"Failed to send resource notification to client: {e}")
    except Exception as e:
        logger.error(f"Error in notify_resource_change: {e}")


async def notify_resource_list_changed() -> None:
    """Send all clients a resource list changed notification."""
    try:
        await store_client_session_if_needed()
        # Create a set to track successfully sent notifications
        sent_sessions = set()
        
        # Keep trying until all sessions are processed
        while True:
            # Get sessions that haven't been sent to yet
            remaining_sessions = set(client_sessions.values()) - sent_sessions
            if not remaining_sessions:
                logger.info("All resource list change notifications have been processed")
                # clear the sent_sessions
                sent_sessions.clear()
                break
                
            for client_session in remaining_sessions:
                try:
                    await client_session.send_notification(
                        ServerNotification(
                            ResourceListChangedNotification(
                                method="notifications/resources/list_changed",
                            )
                        )
                    )
                    logger.info(f"Sent resource list changed notification to client {client_session}")
                    sent_sessions.add(client_session)
                except Exception as e:
                    logger.error(f"Error sending notification to client {client_session}: {e}")
                    # Remove the disconnected session
                    if client_session in client_sessions.values():
                        del client_sessions[client_session]
                    # Continue with other sessions
                    continue
                    
    except Exception as e:
        logger.error(f"Error in notify_resource_list_changed: {e}")

async def notify_tool_list_changed() -> None:
    """Send a tool list changed notification."""
    try:
        await store_client_session_if_needed()
        # Create a set to track successfully sent notifications
        sent_sessions = set()
        
        # Keep trying until all sessions are processed
        while True:
            # Get sessions that haven't been sent to yet
            remaining_sessions = set(client_sessions.values()) - sent_sessions
            if not remaining_sessions:
                logger.info("All tool list change notifications have been processed")
                # clear the sent_sessions
                sent_sessions.clear()
                break
                
            for client_session in remaining_sessions:
                try:
                    await client_session.send_notification(
                        ServerNotification(
                            ToolListChangedNotification(
                                method="notifications/tools/list_changed",
                            )
                        )
                    )
                    logger.info(f"Sent tool list changed notification to client {client_session}")
                    sent_sessions.add(client_session)
                except Exception as e:
                    logger.error(f"Error sending tool list notification to client {client_session}: {e}")
                    # Remove the disconnected session
                    if client_session in client_sessions.values():
                        del client_sessions[client_session]
                    # Continue with other sessions
                    continue
                    
    except Exception as e:
        logger.error(f"Error in notify_tool_list_changed: {e}")

async def notify_prompt_list_changed() -> None:
    """Send a prompt list changed notification."""
    try:
        await store_client_session_if_needed()
        # Create a set to track successfully sent notifications
        sent_sessions = set()
        
        # Keep trying until all sessions are processed
        while True:
            # Get sessions that haven't been sent to yet
            remaining_sessions = set(client_sessions.values()) - sent_sessions
            if not remaining_sessions:
                logger.info("All prompt list change notifications have been processed")
                # clear the sent_sessions
                sent_sessions.clear()
                break
                
            for client_session in remaining_sessions:
                try:
                    await client_session.send_notification(
                        ServerNotification(
                            PromptListChangedNotification(
                                method="notifications/prompts/list_changed",
                            )
                        )
                    )
                    logger.info(f"Sent prompt list changed notification to client {client_session}")
                    sent_sessions.add(client_session)
                except Exception as e:
                    logger.error(f"Error sending prompt list notification to client {client_session}: {e}")
                    # Remove the disconnected session
                    if client_session in client_sessions.values():
                        del client_sessions[client_session]
                    # Continue with other sessions
                    continue
                    
    except Exception as e:
        logger.error(f"Error in notify_prompt_list_changed: {e}")


def get_capabilities() -> ServerCapabilities:
    return ServerCapabilities(
        prompts=None,
        resources=ResourcesCapability(
            subscribe=True,
            listChanged=True
        ),
        tools=ToolsCapability(
            listChanged=True
        ),
        logging=LoggingCapability(),
        experimental={}
    )


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """
    List the tools available to the server
    """
    tools = [
        Tool(
            name="charge_points_locator",
            description="Find EV charging stations near a specified location",
            inputSchema=CHARGE_POINT_LOCATOR_SCHEMA
        ),
        Tool(
            name="ev_trip_planner",
            description="Plan a trip for an electric vehicle",
            inputSchema=EV_TRIP_PLANNER_SCHEMA
        ),
        
    ]
    return tools


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent | ImageContent | EmbeddedResource]:
    try:
        if name == "charge_points_locator":
            address = arguments.get("address")
            max_distance = arguments.get("max_distance")
            socket_type = arguments.get("socket_type")
            result = charge_points_locator(address, max_distance, socket_type)
            return [TextContent(type="text", text=str(result))]
        elif name == "ev_trip_planner":
            user_address = arguments.get("user_address")
            user_destination_address = arguments.get("user_destination_address")
            socket_type = arguments.get("socket_type")
            result = ev_trip_planner(user_address, user_destination_address, socket_type)
            return [TextContent(type="text", text=result)]
        else:
            return [TextContent(type="text", text="Tool not found")]
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return [TextContent(type="text", text=f"Error calling tool {name}: {e}")]


@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    resources = []

    # find all the pdf files in /home/abiorh/ai/mcp_learning/Data
    try:
        pdf_files = glob.glob("/home/abiorh/ai/mcp_learning/Data/*.pdf")
        for pdf_path in pdf_files:
            filename =  os.path.basename(pdf_path)
            name_without_extension = os.path.splitext(filename)[0]
            resources.append(
                Resource(
                    uri=f"file:///pdf/{name_without_extension}",
                    name=name_without_extension.title(),
                    description=f"EV Charging Station User Guide: {name_without_extension.title()}",
                    mime_type="application/pdf"
                )
            )
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
    return resources

@server.read_resource()
async def handle_read_resource(uri: str) -> bytes:
    "handle the read resource request"
    try:
        logger.info(f"Reading resource: {uri}")
        await notify_resource_change(uri)
        time.sleep(3)
        await notify_resource_list_changed()
        time.sleep(3)
        await notify_tool_list_changed()
        time.sleep(3)
        await notify_prompt_list_changed()
        # test the create message
        # result = await server.request_context.session.create_message(
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": {
        #                 "type": "text",
        #                 "text": "Hello, how are you?"
        #             }
        #         }
        #     ],
        #     max_tokens=100,
        #     system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
        #     temperature=0.5,
        #     stop_sequences=["\n\n"],
        #     model_preferences={
        #         "hints":[
        #             {   "provider": "gemini",
        #                 "name": "gemini-1.5-flash",
        #             }
        #         ]
        #     }
        # )
        # logger.info(f"Result from create message: {result}")
        # extract the name from the url
        if not str(uri).startswith("file:///pdf/"):
            raise ValueError(f"Unsupported resource URL: {uri}")
        
        # Get the filename from the URI
        document_name = str(uri).split("/pdf/")[1].lower()
        if "%20" in document_name:
            document_name = document_name.replace("%20", " ")
        
        # Construct the full path
        pdf_path = os.path.join("/home/abiorh/ai/mcp_learning/Data", f"{document_name}.pdf")
        
        # Log the full path for debugging
        logger.info(f"Attempting to read PDF from: {pdf_path}")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            error_msg = f"PDF file not found at: {pdf_path}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
        
        # Check file size
        file_size = os.path.getsize(pdf_path)
        logger.info(f"PDF file size: {file_size} bytes")
        if file_size == 0:
            error_msg = f"PDF file is empty (0 bytes): {pdf_path}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
        
        # Try to load with PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()  # Load and split into pages
        
        # Check if any pages were extracted
        if not pages:
            error_msg = f"PyPDFLoader didn't extract any pages from: {pdf_path}"
            logger.error(error_msg)
            
            # Try alternative method (PyMuPDF/fitz as fallback)
            logger.info("Attempting fallback with PyMuPDF...")
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                pdf_text = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pdf_text += f"\n\n--- Page {page_num+1} ---\n\n"
                    pdf_text += page.get_text()
                
                if not pdf_text.strip():
                    logger.error("Fallback also failed to extract text. This might be a scanned PDF without OCR.")
                    return "ERROR: This appears to be a scanned PDF without text. OCR processing would be required."
                
                logger.info(f"Successfully extracted text with PyMuPDF fallback: {len(pdf_text)} characters")
                return pdf_text
            except ImportError:
                logger.error("PyMuPDF not installed. Cannot use fallback method.")
                return f"ERROR: {error_msg} and PyMuPDF fallback not available."
            except Exception as e:
                logger.error(f"PyMuPDF fallback also failed: {str(e)}")
                return f"ERROR: Failed to extract text with both primary and fallback methods. Error: {str(e)}"
        import uuid
        progress_token = f"charging_station_user_guide_{str(uuid.uuid4())}"
        # Extract text from pages
        pdf_text = ""
        for i, page in enumerate(pages):
            await server.request_context.session.send_progress_notification(progress_token, i, len(pages))
            page_content = page.page_content.strip()
            pdf_text += f"\n\n--- Page {i+1} ---\n\n"
            pdf_text += page_content
        
        # Check if extracted text is empty
        if not pdf_text.strip():
            logger.warning(f"Extracted text is empty for PDF: {pdf_path}")
            return "ERROR: No text could be extracted from this PDF. It might be a scanned document without OCR."
        
        logger.info(f"Successfully extracted text: {len(pdf_text)} characters")
        
        # Notify subscribers about the resource change
        #await notify_resource_change(uri)
        
        return pdf_text

    except Exception as e:
        error_msg = f"Error reading PDF: {str(e)}"
        logger.error(error_msg)
        return f"ERROR: {error_msg}"

PROMPTS = {
    "find-charging-stations": Prompt(
        name="find-charging-stations",
        description="Find nearby EV charging stations",
        arguments=[
            PromptArgument(
                name="location",
                description="Address or location to search for charging stations",
                required=True
            ),
            PromptArgument(
                name="radius",
                description="Search radius in kilometers",
                required=False
            ),
            PromptArgument(
                name="socket_type",
                description="Type of charging socket (e.g., CCS, CHAdeMO, Type 2)",
                required=False
            )
        ],
    ),
    "charging-time-estimate": Prompt(
        name="charging-time-estimate",
        description="Estimate charging time for an EV",
        arguments=[
            PromptArgument(
                name="vehicle_model",
                description="EV make and model",
                required=True
            ),
            PromptArgument(
                name="current_charge",
                description="Current battery percentage",
                required=True
            ),
            PromptArgument(
                name="target_charge",
                description="Desired battery percentage",
                required=True
            ),
            PromptArgument(
                name="charger_power",
                description="Charging station power in kW",
                required=True
            )
        ],
    ),
    "route-planner": Prompt(
        name="route-planner",
        description="Plan a route with EV charging stops",
        arguments=[
            PromptArgument(
                name="start_location",
                description="Starting point address",
                required=True
            ),
            PromptArgument(
                name="end_location",
                description="Destination address",
                required=True
            ),
            PromptArgument(
                name="vehicle_range",
                description="Vehicle's full charge range in kilometers",
                required=True
            ),
            PromptArgument(
                name="current_charge",
                description="Current battery percentage",
                required=False
            )
        ],
    )
}

@server.list_prompts()
async def handle_list_prompts() -> list[Prompt]:
    return list(PROMPTS.values())

@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict | None) -> GetPromptResult:
    if name not in PROMPTS:
        raise ValueError(f"Prompt not found: {name}")
    if name == "find-charging-stations":
        try:
            if not arguments:
                raise ValueError("Location and radius are required for find-charging-stations prompt")
            prompt = FindChargingStationsPrompt(**arguments)
        except ValidationError as e:
            raise ValueError(f"Invalid arguments for find-charging-stations prompt: {e}")
        location = prompt.location
        radius = prompt.radius if prompt.radius else "any"
        socket_type = prompt.socket_type if prompt.socket_type else "any"
        prompt_response = GetPromptResult(
            messages = [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Find nearby EV charging stations near {location} within {radius} km with {socket_type} sockets"
                    )
                )
            ]
        )
        return prompt_response
    elif name == "charging-time-estimate":
        try:
            if not arguments:
                raise ValueError("Vehicle model, current charge, target charge, and charger power are required for charging-time-estimate prompt")
            prompt = ChargingTimeEstimatePrompt(**arguments)
        except ValidationError as e:
            raise ValueError(f"Invalid arguments for charging-time-estimate prompt: {e}")
        vehicle_model = prompt.vehicle_model
        current_charge = prompt.current_charge
        target_charge = prompt.target_charge
        charger_power = prompt.charger_power
        prompt_response = GetPromptResult(
            messages = [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Estimate charging time for {vehicle_model} with {current_charge}% charge to {target_charge}% with a charger of {charger_power} kW"
                    )
                )
            ]
        )
        return prompt_response
    elif name == "route-planner":
        if not arguments:
            raise ValueError("Start location, end location, vehicle range, and current charge are required for route-planner prompt")
        start_location = arguments.get("start_location")
        end_location = arguments.get("end_location")
        vehicle_range = arguments.get("vehicle_range")
        current_charge = arguments.get("current_charge", "unknown")
        prompt_response = GetPromptResult(
            messages = [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Plan a route from {start_location} to {end_location} with a vehicle range of {vehicle_range} km and {current_charge}% charge"
                    )
                )
            ]
        )
        return prompt_response

# initialization options
init_options = InitializationOptions(
                server_name="MCP Low Level EV Assistant Server",
                server_version="1.0.0",
                capabilities=get_capabilities(),
            )
# main function
async def sse_main():
    # Create an SSE transport at an endpoint
    sse = SseServerTransport("/messages/")

    # Define handler function for SSE connections
    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            # Run the server with the streams and initialization options
            await server.run(
                streams[0],  # read stream
                streams[1],  # write stream
                init_options
            )

    # Create Starlette routes for SSE and message handling
    starlette_app = Starlette(
        debug=True,
        routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ])
    
    config = uvicorn.Config(
        starlette_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    
    # Create and run server
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()


async def stdio_main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            init_options
        )

if __name__ == "__main__":
    logger.info("Starting MCP server...")
    # uncomment this to run the sse server
    # asyncio.run(sse_main())
    # uncomment this to run the stdio server
    asyncio.run(stdio_main())
    
    

