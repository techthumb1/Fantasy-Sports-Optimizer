import Head from "next/head";
import Image from "next/image";
import "./globals.css";
import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <Head>
        <title>Fantasy Sports Optimizer</title>
        <meta name="description" content="Optimize your fantasy sports teams with AI-driven insights and models." />
      </Head>
      <body className={`${inter.className} flex flex-col bg-gradient-to-r from-gray-900 to-gray-700 text-white`}>
        <header className="w-full max-w-5xl mx-auto mb-12 flex-shrink-0">
          <nav className="flex justify-between items-center py-4">
            <Image src="/logo.svg" alt="Fantasy Sports Optimizer Logo" width={100} height={30} />
            <ul className="flex space-x-8">
              <li><a href="/about" className="text-lg hover:underline">About</a></li>
              <li><a href="/features" className="text-lg hover:underline">Features</a></li>
              <li><a href="/contact" className="text-lg hover:underline">Contact</a></li>
            </ul>
          </nav>
        </header>

        <main className="flex-grow w-full max-w-5xl mx-auto flex flex-col items-center justify-center p-24">
          {children}
        </main>

        <footer className="w-full max-w-5xl mx-auto mt-auto py-4 border-t border-gray-600">
          <div className="text-center">
            <p className="text-sm opacity-70">
              &copy; {new Date().getFullYear()} Fantasy Sports Optimizer. All rights reserved.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
