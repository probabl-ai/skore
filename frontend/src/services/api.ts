import { DataStore } from "@/models";

const { protocol, hostname } = window.location;
const BASE_URL = `${protocol}//${hostname}:8000/api`;

function getErrorMessage(error: unknown) {
  if (error instanceof Error) return error.message;
  return String(error);
}

function reportError(message: string) {
  console.error(message);
}

export async function fetchAllManderUris(): Promise<string[]> {
  try {
    console.log(BASE_URL);
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
    const r = await fetch(`${BASE_URL}/fake-mandrs/${uri}`);
    if (r.status == 200) {
      const m = await r.json();
      return new DataStore(m.uri, m.payload);
    }
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}
