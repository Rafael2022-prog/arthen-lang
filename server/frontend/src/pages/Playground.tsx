import React, { useEffect, useState } from 'react'
import api from '../api/client'

interface ParseResponse {
  token_count: number
  node_count: number
  ast: Record<string, any>
  backend_mode: string
}

interface CompileResponse {
  compiled_code: string
  target_blockchain: string
  ast: Record<string, any>
  security_report: Record<string, any>
  optimization_metrics: Record<string, any>
}

const Playground: React.FC = () => {
  const [examples, setExamples] = useState<string[]>([])
  const [selectedExample, setSelectedExample] = useState<string>('')
  const [code, setCode] = useState<string>('')
  const [parseResult, setParseResult] = useState<ParseResponse | null>(null)
  const [compileResult, setCompileResult] = useState<CompileResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [target, setTarget] = useState<string>('ethereum')
  const allowedTargets = ['ethereum','solana','cosmos','polkadot','near','move_aptos','cardano'] as const
  
  const normalizeTarget = (t: string): string => {
    const x = t.toLowerCase().trim()
    if (x === 'move' || x === 'aptos' || x === 'sui') return 'move_aptos'
    return x
  }
  
  const inferTargetsFromSource = (src: string): string[] => {
    const results: string[] = []
    // ∆compile_target⟨solana⟩
    const m1 = src.match(/∆compile_target⟨\s*([a-zA-Z0-9_]+)\s*⟩/)
    if (m1 && m1[1]) results.push(normalizeTarget(m1[1]))
    // ∆target_chains: [ethereum, solana]
    const m2 = src.match(/∆target_chains:\s*\[([^\]]+)\]/)
    if (m2 && m2[1]) {
      m2[1].split(',').map(s => s.trim()).forEach(w => results.push(normalizeTarget(w)))
    }
    const uniq = Array.from(new Set(results))
    return uniq.filter(r => allowedTargets.includes(r as any))
  }
  
  // Auto-set target dari source directive
  useEffect(() => {
    const ts = inferTargetsFromSource(code)
    if (ts.length > 0 && ts[0] !== target) {
      setTarget(ts[0])
    }
  }, [code])

  useEffect(() => {
    const init = async () => {
      try {
        const { data } = await api.get('/api/examples')
        setExamples(data.examples || [])
        const initial = (data.examples || [])[0]
        if (initial) {
          setSelectedExample(initial)
          const { data: content } = await api.get(`/api/examples/${initial}`)
          setCode(content.content || '')
        }
      } catch (e) {
        console.error(e)
      }
    }
    init()
  }, [])

  const parse = async () => {
    setLoading(true)
    setError('')
    setCompileResult(null)
    try {
      const { data } = await api.post<ParseResponse>('/api/parse', { source: code, model_mode: 'ml' })
      setParseResult(data)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Parse gagal')
    } finally {
      setLoading(false)
    }
  }

  const compile = async (target: string) => {
    setLoading(true)
    setError('')
    try {
      const { data } = await api.post<CompileResponse>('/api/compile', { source: code, target })
      setCompileResult(data)
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || 'Compile gagal')
    } finally {
      setLoading(false)
    }
  }

  const loadExample = async (name: string) => {
    try {
      const { data } = await api.get(`/api/examples/${name}`)
      setSelectedExample(name)
      setCode(data.content || '')
      setParseResult(null)
      setCompileResult(null)
    } catch (e) {
      console.error(e)
    }
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Playground</h1>
        <div className="flex items-center gap-2">
          <select
            value={selectedExample}
            onChange={(e) => loadExample(e.target.value)}
            className="bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm"
          >
            <option value="">Pilih contoh…</option>
            {examples.map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            className="bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm"
            title="Pilih target blockchain"
          >
            <option value="ethereum">ethereum</option>
            <option value="solana">solana</option>
            <option value="cosmos">cosmos</option>
            <option value="polkadot">polkadot</option>
            <option value="near">near</option>
            <option value="move_aptos">move_aptos</option>
            <option value="cardano">cardano</option>
          </select>
          {/* Tombol cepat per target */}
          <div className="hidden md:flex items-center gap-2">
            {allowedTargets.map((t) => (
              <button
                key={t}
                onClick={() => compile(t)}
                className="px-3 py-2 rounded border border-gray-700 hover:border-gray-500 text-white text-xs"
                disabled={loading}
                title={`Compile cepat → ${t}`}
              >
                {t}
              </button>
            ))}
          </div>
          <button
            onClick={() => parse()}
            className="px-4 py-2 rounded bg-brand-600 hover:bg-brand-500 text-white text-sm"
            disabled={loading}
          >
            {loading ? 'Memproses…' : 'Parse & Analyze'}
          </button>
          <button
            onClick={() => compile(target)}
            className="px-4 py-2 rounded border border-gray-700 hover:border-gray-500 text-white text-sm"
            disabled={loading}
          >
            Compile → {target}
          </button>
        </div>
      </div>

      {/* Editor */}
      <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <label className="text-sm text-gray-400">Kode ARTHEN</label>
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="mt-2 w-full h-80 bg-gray-900 border border-gray-800 rounded p-3 font-mono text-sm"
            spellCheck={false}
          />
        </div>

        <div>
          <label className="text-sm text-gray-400">Hasil</label>
          <div className="mt-2 space-y-4">
            {error && (
              <div className="text-red-400 text-sm">{error}</div>
            )}
            {parseResult && (
              <div className="rounded border border-gray-800 p-3 bg-gray-900/40">
                <div className="text-sm">Tokens: <span className="font-semibold">{parseResult.token_count}</span> · Nodes: <span className="font-semibold">{parseResult.node_count}</span> · Backend: {parseResult.backend_mode}</div>
                <pre className="mt-2 text-xs overflow-auto max-h-64">{JSON.stringify(parseResult.ast, null, 2)}</pre>
              </div>
            )}
            {compileResult && (
              <div className="rounded border border-gray-800 p-3 bg-gray-900/40">
                <div className="text-sm">Target: <span className="font-semibold">{compileResult.target_blockchain}</span></div>
                <div className="mt-2 text-sm">Compiled Code:</div>
                <pre className="mt-1 text-xs overflow-auto max-h-64">{compileResult.compiled_code}</pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Playground