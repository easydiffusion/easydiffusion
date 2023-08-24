from easydiffusion import model_manager, app, server, bucket_manager
from easydiffusion.server import server_api  # required for uvicorn

app.init()

server.init()

# Init the app
model_manager.init()
app.init_render_threads()
bucket_manager.init()

# start the browser ui
app.open_browser()
