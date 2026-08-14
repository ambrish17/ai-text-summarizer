import { useState } from 'react'
import { useRouter } from 'next/router'

export default function Login() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const router = useRouter()

  async function handleSubmit(e) {
    e.preventDefault()
    const res = await fetch(process.env.NEXT_PUBLIC_API_URL + 'token/', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, password}),
    })
    const data = await res.json()
    if (res.ok) {
      // store tokens in localStorage (for demo). For production, consider httpOnly cookie for refresh token.
      localStorage.setItem('access', data.access)
      localStorage.setItem('refresh', data.refresh)
      router.push('/products')
    } else {
      alert(JSON.stringify(data))
    }
  }

  return (
    <div style={{padding: '2rem'}}>
      <h1>Login</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Username</label><br/>
          <input value={username} onChange={e => setUsername(e.target.value)} />
        </div>
        <div>
          <label>Password</label><br/>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} />
        </div>
        <button type="submit">Login</button>
      </form>
    </div>
  )
}
