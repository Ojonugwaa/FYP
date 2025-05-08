import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import Header from "./Header.jsx"
import Footer from "./Footer.jsx"
import Skill from "./Skill.jsx"
import Curriculum from "./curriculum.jsx"

function App() {
  return (
    <>
      <Router>
        {/* Optional: Header appears on all pages */}
        <Header />

        <Routes>
          <Route path="/" element={<Skill />} />
          <Route path="/curriculum" element={<Curriculum />} />
        </Routes>

        {/* Optional: Footer appears on all pages */}
        <Footer />
      </Router>
    </>
  )
}

export default App;
