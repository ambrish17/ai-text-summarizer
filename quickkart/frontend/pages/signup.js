import { useState } from 'react'
import { useRouter } from 'next/router'

export default function Signup() {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const router = useRouter()

  async function handleSubmit(e) {
    e.preventDefault()
    const res = await fetch(process.env.NEXT_PUBLIC_API_URL + 'auth/register/', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({username, email, password}),
    })
    if (res.ok) {
      router.push('/login')
    } else {
      const data = await res.json()
      alert(JSON.stringify(data))
    }
  }

  return (
    <div style={{padding: '2rem'}}>
      <h1>Sign up</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Username</label><br/>
          <input value={username} onChange={e => setUsername(e.target.value)} />
        </div>
        <div>
          <label>Email</label><br/>
          <input value={email} onChange={e => setEmail(e.target.value)} />
        </div>
        <div>
          <label>Password</label><br/>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)} />
        </div>
        <button type="submit">Sign up</button>
      </form>
    </div>
  )
}
