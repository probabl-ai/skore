import type { Layout, Report } from "@/models";
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

export async function fetchReport(): Promise<Report | null> {
  try {
    const r = await fetch(`${BASE_URL}/items`);
    checkResponseStatus(r, 200);
    return await r.json();
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}

export async function putLayout(payload: Layout): Promise<Report | null> {
  try {
    const r = await fetch(`${BASE_URL}/report/layout`, {
      method: "PUT",
      body: JSON.stringify(payload),
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

export async function fetchShareableBlob(layout: Layout) {
  try {
    const r = await fetch(`${BASE_URL}/report/share`, {
      method: "POST",
      body: JSON.stringify(layout),
      headers: {
        "Content-Type": "application/json",
      },
    });
    if (r.status == 200) {
      return await r.blob();
    }
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}
