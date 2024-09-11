import type { Layout, ReportItem } from "@/models";

const { protocol, hostname, port: windowPort } = window.location;
// In the general case we expect the webapp to run at the same port as the API
const port = import.meta.env.DEV ? 22140 : windowPort;
export const BASE_URL = `${protocol}//${hostname}:${port}/api`;

function getErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function reportError(message: string) {
  console.error(message);
}

export async function fetchReport(): Promise<{ [key: string]: ReportItem } | null> {
  try {
    const r = await fetch(`${BASE_URL}/report`);
    if (r.status == 200) {
      return await r.json();
    }
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
