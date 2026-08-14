import Head from 'next/head'
import Link from 'next/link'

export default function Home() {
  return (
    <>
      <Head>
        <title>QuickKart</title>
      </Head>
      <main style={{padding: '2rem'}}>
        <h1>Welcome to QuickKart</h1>
        <p>This is a starter Next.js storefront. Connect to the backend API at /api/</p>
        <p><Link href="/products">Browse products</Link></p>
      </main>
    </>
  )
}
