import React, { useEffect, useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const DocView: React.FC = () => {
  const [searchParams] = useSearchParams()
  const path = searchParams.get('path')
  const [content, setContent] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      if (!path) {
        setError('Parameter path tidak ditemukan.')
        setLoading(false)
        return
      }
      try {
        const res = await fetch(`/docs/${path}`)
        if (!res.ok) {
          throw new Error(`Gagal memuat dokumen: ${res.status}`)
        }
        const text = await res.text()
        setContent(text)
      } catch (e: any) {
        setError(e?.message || 'Tidak dapat memuat dokumen.')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [path])

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-4">
        <Link to="/docs" className="text-sm text-gray-300 hover:text-white">← Kembali ke daftar dokumen</Link>
      </div>
      {loading && <div className="text-sm text-gray-400">Memuat dokumen…</div>}
      {error && <div className="text-sm text-red-400">{error}</div>}
      {!loading && !error && (
        <article className="prose prose-invert max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </article>
      )}
    </div>
  )
}

export default DocView