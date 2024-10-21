import type { Layout, Project } from "@/models";
import { useToastsStore } from "@/stores/toasts";

const { protocol, hostname, port: windowPort } = window.location;
// In the general case we expect the webapp to run at the same port as the API
const port = import.meta.env.DEV ? 22140 : windowPort;
export const BASE_URL = `${protocol}//${hostname}:${port}/api`;

function getErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function reportError(message: string) {
  const toastsStore = useToastsStore();
  toastsStore.addToast(message, "error");
  console.error(message);
}

function checkResponseStatus(r: Response, attendedStatusCode: number) {
  if (r.status !== attendedStatusCode) {
    throw new Error(`Server responded with unexpected status code: ${r.status}`);
  }
}

export async function fetchProject(): Promise<Project | null> {
  try {
    const r = await fetch(`${BASE_URL}/project/items`);
    checkResponseStatus(r, 200);
    return await r.json();
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}

export async function putView(view: string, layout: Layout): Promise<Project | null> {
  try {
    const r = await fetch(`${BASE_URL}/project/views?key=${view}`, {
      method: "PUT",
      body: JSON.stringify(layout),
      headers: {
        "Content-Type": "application/json",
      },
    });
    checkResponseStatus(r, 201);
    return await r.json();
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}

export async function deleteView(view: string) {
  try {
    const r = await fetch(`${BASE_URL}/project/views?key=${view}`, {
      method: "DELETE",
    });
    checkResponseStatus(r, 202);
  } catch (error) {
    reportError(getErrorMessage(error));
  }
}
