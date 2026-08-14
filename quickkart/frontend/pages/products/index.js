import { useEffect, useState } from 'react'
import Link from 'next/link'

export default function Products() {
  const [products, setProducts] = useState([])

  useEffect(() => {
    async function load() {
      const res = await fetch(process.env.NEXT_PUBLIC_API_URL + 'products/')
      const data = await res.json()
      setProducts(data)
    }
    load()
  }, [])

  return (
    <div style={{padding: '2rem'}}>
      <h1>Products</h1>
      <p><Link href="/">Home</Link> | <Link href="/cart">Cart</Link></p>
      <ul>
        {products.map(p => (
          <li key={p.id} style={{marginBottom: '1rem'}}>
            <strong>{p.title}</strong><br/>
            <span>{p.description}</span><br/>
            <span>${p.price}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
