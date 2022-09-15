import {
  Link,
  MakeGenerics,
  Outlet,
  ReactLocation,
  Router,
  useMatch,
} from "@tanstack/react-location";


import Home from "./components/layouts/Home";
import Settings from "./components/layouts/Settings";

const location = new ReactLocation();


function App() {
  return (
    <Router
      location={location}
      routes={[
        {path: "/", element: <Home />},
        {path: "settings", element: <Settings />},
      ]}
    >
    </Router>
  );
}

export default App;