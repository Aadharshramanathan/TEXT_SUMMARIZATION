import React from 'react'
import {Link} from 'react-router-dom'
import {className} from "./Navbar.css"

function Navbar() {
    return (
        <div className='navbar'>
            <Link to="/"><button>Home</button></Link>
            <Link to="/single"><button>Single Docs</button></Link>
        </div>
    )
}

export default Navbar
