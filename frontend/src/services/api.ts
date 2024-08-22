import { DataStore, type Layout } from "@/models";

const { protocol, hostname } = window.location;
export const BASE_URL = `${protocol}//${hostname}:8000/api`;

function getErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function reportError(message: string) {
  console.error(message);
}

export async function fetchAllManderUris(): Promise<string[]> {
  try {
    const r = await fetch(`${BASE_URL}/mandrs`);
    const uris = await r.json();
    return uris;
  } catch (error) {
    reportError(getErrorMessage(error));
    return [];
  }
}

export async function fetchMander(uri: string): Promise<DataStore | null> {
  try {
    const r = await fetch(`${BASE_URL}/mandrs/${uri}`);
    if (r.status == 200) {
      const m = await r.json();
      return new DataStore(m.uri, m.payload, m.layout);
    }
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}

export async function putLayout(uri: string, payload: Layout): Promise<DataStore | null> {
  try {
    const r = await fetch(`${BASE_URL}/mandrs${uri}/layout`, {
      method: "PUT",
      body: JSON.stringify(payload),
      headers: {
        "Content-Type": "application/json",
      },
    });
    if (r.status == 201) {
      const m = await r.json();
      return new DataStore(m.uri, m.payload, m.layout);
    }
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}

export async function fetchShareableBlob(uri: string) {
  try {
    const r = await fetch(`${BASE_URL}/stores/share${uri}`);
    if (r.status == 200) {
      return await r.blob();
    }
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}
