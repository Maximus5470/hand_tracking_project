import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Spline 3D ThreeJS Interactive Studio | Next.js App",
  description: "Interactive 3D WebGL Canvas powered by Next.js App Router, @splinetool/react-spline, ThreeJS, and cyber glassmorphism UI.",
  keywords: ["Spline", "ThreeJS", "React 3D", "WebGL", "Next.js", "Cyberpunk UI", "Glassmorphism"],
  authors: [{ name: "Antigravity 3D Engine" }],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark h-full antialiased selection:bg-cyan-500 selection:text-black">
      <body className="h-dvh overflow-hidden bg-[#040810] text-slate-100 font-sans">
        {children}
      </body>
    </html>
  );
}
