import { useEffect, useState } from "react";

// Client-side routing with real URLs. The server answers every non-API path with index.html, so reload and deep links work.
export const ROUTES = { "/": "search", "/how": "how", "/evaluation": "evaluation" };

const CHANGE_EVENT = "pathchange";

export function routeOf(pathname) {
  return ROUTES[pathname.replace(/\/+$/, "") || "/"] ?? "search";
}

export function navigate(path) {
  if (window.location.pathname === path) return;
  window.history.pushState({}, "", path);
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

export function useRoute() {
  const [route, setRoute] = useState(() => routeOf(window.location.pathname));
  useEffect(() => {
    const update = () => setRoute(routeOf(window.location.pathname));
    window.addEventListener("popstate", update);
    window.addEventListener(CHANGE_EVENT, update);
    return () => {
      window.removeEventListener("popstate", update);
      window.removeEventListener(CHANGE_EVENT, update);
    };
  }, []);
  return route;
}

export function followLink(event, path) {
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0) return;
  event.preventDefault();
  navigate(path);
}
