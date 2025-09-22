import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import About from "./pages/About";
import Contact from "./pages/Contact";
import Login from "./pages/Login";
import Orb from "./components/Orb";
import SplashCursor from "./components/ui/SplashCursor";

function App() {
  return (
    <Router>
      <SplashCursor />
      <div style={{ position: "relative", minHeight: "100vh" }}>
        {/* Orb background */}
        <div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            zIndex: 0,
            overflow: "hidden",
            pointerEvents: "auto",
          }}
        >
          <Orb
            hoverIntensity={0.7}
            rotateOnHover={true}
            hue={0}
            forceHoverState={false}
          />
        </div>

        {/* Foreground content */}
        <div style={{ position: "relative", zIndex: 1 }}>
          <Navbar />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/login" element={<Login />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
