from easydiffusion import model_manager, app, server
from easydiffusion.server import server_api  # required for uvicorn

server.init()

# Init the app
model_manager.init()
app.init()
app.init_render_threads()

# start the browser ui
app.open_browser()
