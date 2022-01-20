import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'

import Home from './pages/Home';
import Single_inp from './pages/Single_inp';


function App() {
  return (
    <Router>
      <Routes>
        <Route exact path="/" element={<Home/>}/>
        <Route path="/single" element={<Single_inp/>}/>
      </Routes>
    </Router>
  );
}

export default App;
