import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

interface DocsIndex {
  files: string[]
}

const groupByFolder = (files: string[]) => {
  const groups: Record<string, string[]> = {}
  files.forEach((f) => {
    const parts = f.split('/')
    const folder = parts.length > 1 ? parts.slice(0, -1).join('/') : 'Root'
    if (!groups[folder]) groups[folder] = []
    groups[folder].push(f)
  })
  return groups
}

const defaultFiles = [
  'API.md',
  'CLI.md',
  'DEPRECATION_POLICY.md',
  'GETTING_STARTED.md',
  'HARDENING_GUIDE.md',
  'ICON_VERIFICATION_CHECKLIST.md',
  'INSTALL.md',
  'RELEASE_POLICY.md',
  'SIGNING_AND_NOTARIZATION.md',
  'VERSIONING.md',
  'TUTORIALS/ethereum.md',
  'TUTORIALS/foundry.md',
  'TUTORIALS/solana.md',
]

const Docs: React.FC = () => {
  const [files, setFiles] = useState<string[]>(defaultFiles)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchIndex = async () => {
      try {
        const res = await fetch('/docs/index.json')
        if (res.ok) {
          const data: DocsIndex = await res.json()
          if (Array.isArray(data.files) && data.files.length > 0) {
            setFiles(data.files)
          }
        }
      } catch (e) {
        // ignore, fallback to defaultFiles
      } finally {
        setLoading(false)
      }
    }
    fetchIndex()
  }, [])

  const groups = groupByFolder(files)

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <h1 className="text-2xl font-semibold mb-4">Dokumentasi</h1>
      <p className="text-gray-400 mb-6">Berikut adalah seluruh dokumen dari folder <code>docs/</code>. Klik untuk melihat isi.</p>
      {loading && (
        <div className="text-sm text-gray-400 mb-4">Memuat indeks dokumentasi…</div>
      )}
      <div className="space-y-8">
        {Object.entries(groups).map(([folder, items]) => (
          <section key={folder}>
            <h2 className="text-xl font-medium mb-3">{folder}</h2>
            <ul className="grid sm:grid-cols-2 md:grid-cols-3 gap-3">
              {items.map((f) => {
                const name = f.split('/').pop()!
                const encoded = encodeURIComponent(f)
                return (
                  <li key={f}>
                    <Link to={`/docs/view?path=${encoded}`} className="block px-3 py-2 bg-gray-900 hover:bg-gray-800 rounded border border-gray-800">
                      <div className="font-mono text-sm">{name}</div>
                      <div className="text-xs text-gray-400">{folder === 'Root' ? '' : folder}</div>
                    </Link>
                  </li>
                )
              })}
            </ul>
          </section>
        ))}
      </div>
    </div>
  )
}

export default Docs