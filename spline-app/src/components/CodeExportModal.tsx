'use client';

import { useState } from 'react';
import { X, Copy, Check, Terminal, FileCode } from 'lucide-react';

interface CodeExportModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function CodeExportModal({ isOpen, onClose }: CodeExportModalProps) {
  const [activeTab, setActiveTab] = useState<'next' | 'react'>('next');
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const nextCode = `import Spline from '@splinetool/react-spline/next';

export default function Home() {
  return (
    <main className="w-full h-screen">
      <Spline
        scene="https://prod.spline.design/r5KAc7jXVA7ryXus/scene.splinecode" 
      />
    </main>
  );
}`;

  const viteCode = `import Spline from '@splinetool/react-spline';

export function App() {
  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <Spline scene="https://prod.spline.design/r5KAc7jXVA7ryXus/scene.splinecode" />
    </div>
  );
}`;

  const codeToCopy = activeTab === 'next' ? nextCode : viteCode;

  const handleCopy = () => {
    navigator.clipboard.writeText(codeToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-xl animate-in fade-in duration-200">
      <div className="relative w-full max-w-2xl rounded-2xl glass-hud border border-cyan-500/30 p-6 shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-slate-800">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <Terminal className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-slate-100">Spline Integration Snippet</h3>
              <p className="text-xs text-slate-400 font-mono">Copy and paste directly into your project</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Framework Tabs */}
        <div className="flex items-center space-x-2 mb-4">
          <button
            onClick={() => setActiveTab('next')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-mono transition-all cursor-pointer ${
              activeTab === 'next'
                ? 'bg-cyan-500/20 border border-cyan-400 text-cyan-300 shadow-[0_0_10px_rgba(6,182,212,0.3)]'
                : 'bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileCode className="w-4 h-4 text-cyan-400" />
            <span>Next.js App Router</span>
          </button>
          <button
            onClick={() => setActiveTab('react')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-mono transition-all cursor-pointer ${
              activeTab === 'react'
                ? 'bg-purple-500/20 border border-purple-400 text-purple-300 shadow-[0_0_10px_rgba(168,85,247,0.3)]'
                : 'bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <FileCode className="w-4 h-4 text-purple-400" />
            <span>Vite / React SPA</span>
          </button>
        </div>

        {/* Code Snippet Box */}
        <div className="relative rounded-xl bg-slate-950 border border-slate-800 p-4 font-mono text-xs text-slate-200 overflow-x-auto">
          <button
            onClick={handleCopy}
            className="absolute top-3 right-3 flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-xs text-slate-300 hover:text-white transition-all cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400 font-medium">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-cyan-400" />
                <span>Copy Code</span>
              </>
            )}
          </button>
          <pre className="pr-24 leading-relaxed text-cyan-100">
            <code>{codeToCopy}</code>
          </pre>
        </div>

        {/* Installation Instruction */}
        <div className="mt-4 p-3 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between text-xs font-mono text-slate-400">
          <span>Required NPM Package:</span>
          <code className="text-cyan-300 bg-slate-950 px-2 py-1 rounded border border-slate-800">
            npm i @splinetool/react-spline
          </code>
        </div>
      </div>
    </div>
  );
}
