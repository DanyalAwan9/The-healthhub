export default function ChatBubble({ role, content, typing = false }) {
  const isUser = role === 'user';
  return (
    <div className={`msg msg--${isUser ? 'user' : 'bot'}`}>
      <div className="msg__meta">{isUser ? 'You' : 'HealthHub'}</div>
      {typing ? (
        <span className="typing"><i /><i /><i /></span>
      ) : (
        <div>
          {String(content || '').split('\n').map((line, i, arr) => (
            <span key={i}>
              {line}
              {i < arr.length - 1 && <br />}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
