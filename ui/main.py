from easydiffusion import model_manager, app, server
from easydiffusion.server import server_api  # required for uvicorn

# Init the app
model_manager.init()
app.init()
server.init()

# start the browser ui
app.open_browser()

app.init_render_threads()
