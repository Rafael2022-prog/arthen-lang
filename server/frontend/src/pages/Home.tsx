import React from 'react'
import { Link } from 'react-router-dom'

const features = [
  { title: 'AI‑Native Parsing', desc: 'Neural lexer & transformer AST generator yang dioptimasi untuk mesin.' },
  { title: 'ML Consensus Harmony', desc: 'Konsensus multi‑algoritma yang sinkron untuk integrasi AI/ML.' },
  { title: 'Multi‑Chain Compilation', desc: 'Compile ke Ethereum, Solana, Cosmos, Polkadot, NEAR, Cardano, dll.' },
  { title: 'Cross‑Chain Bridge', desc: 'Jembatan AI untuk interoperabilitas lintas rantai dengan keamanan neural.' },
  { title: 'Security Analysis', desc: 'Analisis keamanan berbasis ML dengan laporan otomatis.' },
  { title: 'Optimizations', desc: 'AI‑enhanced codegen, gas optimization, dan performance tuning.' },
]

const Home: React.FC = () => {
  return (
    <div>
      {/* Hero */}
      <section className="bg-hero relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 py-20 text-center">
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight">
            Bahasa Pemrograman AI‑Native untuk Blockchain
          </h1>
          <p className="mt-6 text-lg text-gray-300 max-w-3xl mx-auto">
            ARTHEN‑LANG memadukan AI, ML consensus, dan kompilasi multi‑chain. Bangun kontrak pintar generasi berikutnya dengan kemampuan AI terintegrasi.
          </p>
          <div className="mt-10 flex items-center justify-center gap-4">
            <Link to="/playground" className="px-6 py-3 rounded-lg bg-brand-600 hover:bg-brand-500 text-white font-medium">
              Coba Playground
            </Link>
            <a href="https://github.com/Rafael2022-prog/ARTHEN-LANG/releases" target="_blank" rel="noreferrer" className="px-6 py-3 rounded-lg border border-gray-700 hover:border-gray-500 text-white font-medium">
              Lihat Rilis
            </a>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-16">
          <h2 className="text-2xl md:text-3xl font-bold">Fitur Utama</h2>
          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((f) => (
              <div key={f.title} className="rounded-xl border border-gray-800 bg-gray-900/40 p-6 hover:border-gray-700">
                <h3 className="text-lg font-semibold">{f.title}</h3>
                <p className="mt-2 text-sm text-gray-300">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-4 py-12 text-center">
          <p className="text-gray-300">Semua <em>35 tests</em> telah lulus di repo ini. Mulai eksplorasi ARTHEN‑LANG sekarang.</p>
          <div className="mt-6">
            <Link to="/playground" className="px-6 py-3 rounded-lg bg-brand-600 hover:bg-brand-500 text-white font-medium">Buka Playground</Link>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Home