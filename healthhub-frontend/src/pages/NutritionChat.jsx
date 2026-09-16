import { useEffect, useRef, useState } from 'react';
import Icon from '../components/Icon';
import ChatBubble from '../components/ChatBubble';
import { useUser } from '../context/UserContext';
import { useToast } from '../context/ToastContext';
import { getChatHistory, nutritionChat, clearChat } from '../api/client';
import { MOCK_CHAT_HISTORY, mockChatReply } from '../api/mockData';

const SUGGESTIONS = [
  "What's a cheap high-protein breakfast under PKR 200?",
  'Is brown bread better than roti?',
  "I'm always hungry on my diet",
];

const GREETING = (hasProfile) => ({
  role: 'bot',
  content: hasProfile
    ? 'Hi! Ask me about calories, Pakistani foods, swaps or grocery prices.'
    : 'Hi! Ask away — create a profile on the Profile page and I can tailor answers to your targets.',
});

export default function NutritionChat() {
  const { user } = useUser();
  const { show } = useToast();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [live, setLive] = useState(true);
  const logRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const { data } = await getChatHistory(user?.name);
        if (cancelled) return;
        const msgs = data?.messages || [];
        setMessages(msgs.length ? msgs : [GREETING(!!user)]);
        setLive(true);
      } catch {
        if (!cancelled) { setMessages(user ? MOCK_CHAT_HISTORY : [GREETING(false)]); setLive(false); }
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.name]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [messages, sending]);

  const send = async (e) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || sending) return;
    setMessages((m) => [...m, { role: 'user', content: q }]);
    setInput('');
    setSending(true);
    try {
      const { data } = await nutritionChat(q, user?.name);
      setMessages((m) => [...m, { role: 'bot', content: data.content }]);
    } catch {
      setMessages((m) => [...m, { role: 'bot', content: mockChatReply(q) }]);
    } finally {
      setSending(false);
    }
  };

  const onClear = async () => {
    if (!window.confirm("Clear this conversation? This can't be undone.")) return;
    try { await clearChat(user?.name); } catch { /* offline: clear locally anyway */ }
    setMessages([GREETING(!!user)]);
    show('Chat cleared.', 'ok');
  };

  return (
    <section className="page">
      <div className="page__head">
        <div className="row-between">
          <div>
            <h1>AI Nutritionist</h1>
            <p className="muted">Ask about calories, Pakistani foods, swaps and grocery prices.</p>
          </div>
          <button className="btn btn-outline btn-sm" type="button" onClick={onClear}>
            <Icon name="trash" className="ic--sm" /> Clear chat
          </button>
        </div>
      </div>
      {!live && (
        <div className="mb-3">
          <span className="badge badge--info"><Icon name="alert" className="ic--sm" /> Demo data — backend API not connected</span>
        </div>
      )}

      <div className="chat">
        <div className="chat__log" ref={logRef}>
          {messages.map((m, i) => <ChatBubble key={i} role={m.role} content={m.content} />)}
          {sending && <ChatBubble role="bot" typing />}
        </div>
        <div className="chat__suggest">
          {SUGGESTIONS.map((s) => (
            <button key={s} type="button" className="chip" onClick={() => setInput(s)}>{s}</button>
          ))}
        </div>
        <form className="chat__composer" onSubmit={send}>
          <input
            className="input" autoComplete="off" placeholder="Type a message…" aria-label="Message"
            value={input} onChange={(e) => setInput(e.target.value)}
          />
          <button className="chat__send" type="submit" aria-label="Send message" disabled={sending || !input.trim()}>
            <Icon name="send" />
          </button>
        </form>
      </div>
    </section>
  );
}
