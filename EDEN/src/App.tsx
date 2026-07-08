import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Game from './Game';
import About from './pages/About';
import SnnDashboard from './pages/SnnDashboard';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Game />} />
        <Route path="/about" element={<About />} />
        <Route path="/snn-dashboard" element={<SnnDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
