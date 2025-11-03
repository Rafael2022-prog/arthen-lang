import React, { useEffect, useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkBreaks from 'remark-breaks'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'

// Daftar tag HTML yang umum/diizinkan agar tidak di-escape
const allowedSimpleHtmlTags = new Set([
  'img','br','hr','p','div','span','a','h1','h2','h3','h4','h5','h6','ul','ol','li','code','pre','strong','em','table','thead','tbody','tr','td','th','blockquote','sup','sub','del','ins','mark','small','b','i'
])

const sanitizePath = (raw: string | null): string | null => {
  if (!raw) return null
  const trimmed = raw.trim().replace(/^\/+|\/+$/g, '')
  if (!trimmed) return null
  // Blok path berbahaya
  if (trimmed.includes('..') || trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
    return null
  }
  return trimmed
}

// Normalisasi konten markdown untuk kasus umum yang sering membuat tampilan "rusak"
// - Sisipkan newline setelah penutup tag HTML block (</p>, </div>, dll) bila diikuti header '#'
// - Hilangkan spasi di awal baris sebelum header '#'
// - Escape pseudo-tag seperti <bool>, <int>, <string> agar ditampilkan sebagai teks, bukan HTML element
// - Samakan newline ke \n
const normalizeMarkdown = (text: string): string => {
  let t = text.replace(/\r\n/g, '\n')
  // Jika ada literal HTML diikuti '#' pada baris yang sama, pecah menjadi baris baru
  t = t.replace(/>(\s*)#(?=\s)/g, '>\n\n#')
  // Hilangkan spasi di awal sebelum tanda '#'
  t = t.replace(/^\s+#/gm, '#')
  // Escape pseudo-tag sederhana yang tidak memiliki atribut: <tag> atau </tag>
  t = t.replace(/<\/?([a-z][a-z0-9_-]*)>/g, (m, tag: string) => {
    // Jika tag umum HTML, biarkan apa adanya
    if (allowedSimpleHtmlTags.has(tag)) return m
    // Selain itu, render sebagai teks agar tidak jadi custom element atau dihapus sanitizer
    return m.replace('<', '&lt;').replace('>', '&gt;')
  })
  return t
}

export default function DocView() {
  const [params] = useSearchParams()
  const [content, setContent] = useState<string>('')
  const [error, setError] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(true)

  const safePath = sanitizePath(params.get('path'))

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
    return () => controller.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safePath])

  if (loading) {
    return <div className="p-4 text-sm">Memuat dokumen...</div>
  }

  if (error) {
    return (
      <div className="p-4 text-sm text-red-600">
        Error: {error} — kembali ke <Link className="underline" to="/docs">Docs</Link>
      </div>
    )
  }

  return (
    <div className="prose max-w-none p-4">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkBreaks]}
        rehypePlugins={[rehypeRaw, rehypeSanitize]}
        components={{
          a: (props) => <a {...props} target="_blank" rel="noopener noreferrer" />,
          img: (props) => {
            // Pastikan path relatif bekerja dari public/
            const src = props.src || ''
            const normalizedSrc = src.startsWith('/') ? src : `/${src}`
            return <img {...props} src={normalizedSrc} style={{ maxWidth: '100%' }} />
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}