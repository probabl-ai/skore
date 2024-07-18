import { type DataStore } from "../models";

const BASE_URL = "http://localhost:8000/api";

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
      return m as DataStore;
    }
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}
