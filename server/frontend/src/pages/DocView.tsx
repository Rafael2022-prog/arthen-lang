import React, { useEffect, useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'

const sanitizePath = (raw: string): string => {
  let p = raw
  try { p = decodeURIComponent(p) } catch {}
  // Normalisasi pemisah path Windows ke POSIX
  p = p.replace(/\\/g, '/')
  // Buang drive letter Windows (mis. C:/, R:/)
  p = p.replace(/^[A-Za-z]:\//, '')
  // Jika masih ada prefix menuju folder docs, buang segala sesuatu sebelum 'docs/'
  p = p.replace(/.*?docs\//i, '')
  // Buang prefix '/docs/' jika ada
  p = p.replace(/^\/?docs\//i, '')
  // Buang leading slash
  p = p.replace(/^\//, '')
  // Hindari traversal
  if (p.includes('..')) {
    return ''
  }
  return p
}

// Normalisasi konten markdown untuk kasus umum yang sering membuat tampilan "rusak"
// - Sisipkan newline setelah penutup tag HTML block (</p>, </div>, dll) bila diikuti header '#'
// - Hilangkan spasi di awal baris sebelum header '#'
// - Samakan newline ke \n
const normalizeMarkdown = (text: string): string => {
  let t = text.replace(/\r\n/g, '\n')
  // Jika ada literal HTML diikuti '#' pada baris yang sama, pecah menjadi baris baru
  t = t.replace(/>(\s*)#(?=\s)/g, '>\n\n#')
  // Hilangkan spasi di awal sebelum tanda '#'
  t = t.replace(/^\s+#/gm, '#')
  return t
}

const DocView: React.FC = () => {
  const [searchParams] = useSearchParams()
  const rawPath = searchParams.get('path') || ''
  const safePath = sanitizePath(rawPath)
  const [content, setContent] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()

    const load = async () => {
      if (!safePath) {
        setError('Path dokumen tidak valid. Pastikan menggunakan path relatif dari folder docs/, mis. API.md atau TUTORIALS/solana.md')
        setLoading(false)
        return
      }
      try {
        const url = `/docs/${encodeURI(safePath)}`
        const res = await fetch(url, { signal: controller.signal })
        if (!res.ok) {
          throw new Error(`Gagal memuat dokumen: ${res.status}`)
        }
        const text = await res.text()
        // Deteksi fallback HTML (SPA) agar tidak dirender sebagai markdown
        const lower = text.slice(0, 200).toLowerCase()
        if (lower.includes('<!doctype html>') || lower.includes('<html') || lower.includes('/@vite/client')) {
          throw new Error('Dokumen tidak ditemukan dalam /docs/. Pastikan path relatif dan file ada di server/frontend/public/docs')
        }
        const normalized = normalizeMarkdown(text)
        setContent(normalized)
      } catch (e: any) {
        const msg = String(e?.message || '').toLowerCase()
        const name = String(e?.name || '').toLowerCase()
        // Abaikan AbortError / net::ERR_ABORTED yang terjadi saat HMR/strict mode double-invoke
        if (name.includes('abort') || msg.includes('abort')) {
          return
        }
        setError(e?.message || 'Tidak dapat memuat dokumen.')
      } finally {
        setLoading(false)
      }
    }
    load()

    return () => {
      controller.abort()
    }
  }, [safePath])

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-4">
        <Link to="/docs" className="text-sm text-gray-300 hover:text-white">← Kembali ke daftar dokumen</Link>
      </div>
      {loading && <div className="text-sm text-gray-400">Memuat dokumen…</div>}
      {error && <div className="text-sm text-red-400">{error}</div>}
      {!loading && !error && (
        <article className="prose prose-invert max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]} rehypePlugins={[rehypeRaw, rehypeSanitize]}>{content}</ReactMarkdown>
        </article>
      )}
    </div>
  )
}

export default DocView