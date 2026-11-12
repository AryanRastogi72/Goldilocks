// Thin client for the EvidenceLens API. Every failure becomes an ApiFail with a readable message.
export class ApiFail extends Error {
  constructor(message, code, status) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiFail("The search server is not reachable.", "network", 0);
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const error = body && body.error;
    throw new ApiFail(error ? error.message : `The server answered with status ${response.status}.`, error ? error.code : "http_error", response.status);
  }
  return body;
}

const post = (path, payload) =>
  request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

export const health = () => request("/api/health");
export const search = (payload) => post("/api/search", payload);
export const clarify = (payload) => post("/api/clarify", payload);
export const evaluation = () => request("/api/eval");
export const documentFor = (extId, sessionId) =>
  request(`/api/doc/${encodeURIComponent(extId)}${sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : ""}`);
