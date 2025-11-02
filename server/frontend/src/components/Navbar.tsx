import React from 'react'
import { Link, NavLink } from 'react-router-dom'

const Navbar: React.FC = () => {
  return (
    <header className="sticky top-0 z-50 bg-gray-950/80 backdrop-blur border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2">
          <span className="text-brand-400 text-2xl">∇⟨</span>
          <span className="font-semibold tracking-wide">ARTHEN‑LANG</span>
        </Link>
        <nav className="flex items-center gap-6 text-sm">
          <NavLink to="/" className={({isActive}) => isActive ? 'text-white' : 'text-gray-300 hover:text-white'}>Home</NavLink>
          <NavLink to="/playground" className={({isActive}) => isActive ? 'text-white' : 'text-gray-300 hover:text-white'}>Playground</NavLink>
          <a href="https://github.com/Rafael2022-prog/ARTHEN-LANG" target="_blank" rel="noreferrer" className="text-gray-300 hover:text-white">GitHub</a>
          <a href="/docs" className="text-gray-300 hover:text-white">Docs</a>
        </nav>
      </div>
    </header>
  )
}

export default Navbar