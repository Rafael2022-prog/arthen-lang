import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Playground from './pages/Playground'

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/playground" element={<Playground />} />
        </Routes>
      </main>
      <footer className="mt-16 border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-8 text-sm text-gray-400 flex items-center justify-between">
          <div>© {new Date().getFullYear()} ARTHEN‑LANG</div>
          <div className="space-x-4">
            <a href="https://github.com/Rafael2022-prog/ARTHEN-LANG/releases" className="hover:text-white">Rilis</a>
            <a href="https://github.com/Rafael2022-prog/ARTHEN-LANG" className="hover:text-white">GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App