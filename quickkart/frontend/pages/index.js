import Head from 'next/head'

export default function Home() {
  return (
    <>
      <Head>
        <title>QuickKart</title>
      </Head>
      <main style={{padding: '2rem'}}>
        <h1>Welcome to QuickKart</h1>
        <p>This is a starter Next.js storefront. Connect to the backend API at /api/</p>
      </main>
    </>
  )
}
