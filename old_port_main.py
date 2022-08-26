from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get('/', response_class=HTMLResponse)
def read_root():
    return '''
    <style>
    body {
        font-family: Arial;
        font-size: 11pt;
    }
    </style>
    <h4>The UI has moved to <a href="http://localhost:9000">http://localhost:9000</a>. The current address that you used (ending with :8000) will be removed in the future, so please use <a href="http://localhost:9000">http://localhost:9000</a> going ahead (and in any bookmarks you've saved).</h4>

    <h3>Why has the address changed?</h3>
    <p>The previously used port (8000) is often used by other servers, which results in port conflicts. So the project's port number has been changed, while the project is still young. Otherwise port-conflicts with 8000 will be a common source of new-user issues in the future.</p>
    <p>Sorry about this, and apologies for the inconvenience :)</p>
    '''
