import type { ActivityFeedDto } from "@/dto";
import { getErrorMessage } from "@/services/utils";
import { useToastsStore } from "@/stores/toasts";

const { protocol, hostname, port: windowPort } = window.location;
// In the general case we expect the webapp to run at the same port as the API
const port = import.meta.env.DEV ? 22140 : windowPort;
export const BASE_URL = `${protocol}//${hostname}:${port}/api`;

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

export async function fetchActivityFeed(after?: string): Promise<ActivityFeedDto | null> {
  try {
    let url = `${BASE_URL}/project/activity`;
    if (after) {
      url += `?after=${after}`;
    }
    const r = await fetch(url);
    checkResponseStatus(r, 200);
    return await r.json();
  } catch (error) {
    reportError(getErrorMessage(error));
  }
  return null;
}

export async function setNote(
  key: string,
  message: string,
  version: number = -1
): Promise<object | null> {
  try {
    const r = await fetch(`${BASE_URL}/project/note`, {
      method: "PUT",
      body: JSON.stringify({ key, message, version }),
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
