import { acceptHMRUpdate, defineStore } from "pinia";
import { shallowRef } from "vue";

import { deserializeProjectItemDto, type PresentableItem } from "@/models";
import { fetchActivityFeed, setNote } from "@/services/api";
import { poll } from "@/services/utils";

type ActivityPresentableItem = PresentableItem & { icon: string };

export const useActivityStore = defineStore("activity", () => {
  // this object is not deeply reactive as it may be very large
  const items = shallowRef<ActivityPresentableItem[]>([]);

  /**
   * Set a note on a currently displayed note
   */
  async function setNoteOnItem(key: string, version: number, message: string) {
    // const updateIndex = getCurrentItemUpdateIndex(key);
    await setNote(key, message, version);
  }

  /**
   * Fetch project data from the backend.
   */
  let _isCanceledCall = false;
  let _lastFetchTime = new Date(1, 1, 1, 0, 0, 0, 0);
  async function _fetch() {
    if (!_isCanceledCall) {
      const now = new Date();
      const feed = await fetchActivityFeed(_lastFetchTime.toISOString());
      _lastFetchTime = now;
      if (feed !== null) {
        const newItems = feed.map((i) => ({
          ...deserializeProjectItemDto(i),
          icon: i.media_type.startsWith("text") ? "icon-pill" : "icon-playground",
        }));
        items.value = [...newItems, ...items.value];
      }
    }
  }
  /**
   * Start real time sync with the server.
   */
  let _stopBackendPolling: (() => void) | null = null;
  async function startBackendPolling() {
    // ensure that there is only one polling running
    if (_stopBackendPolling === null) {
      _isCanceledCall = false;
      _stopBackendPolling = await poll(_fetch, 1500);
    }
  }

  /**
   * Stop real time sync with the server.
   */
  function stopBackendPolling() {
    _isCanceledCall = true;
    if (_stopBackendPolling) {
      _stopBackendPolling();
      _stopBackendPolling = null;
    }
  }

  return {
    // refs
    items,
    // actions
    setNoteOnItem,
    startBackendPolling,
    stopBackendPolling,
  };
});

if (import.meta.hot) {
  import.meta.hot.accept(acceptHMRUpdate(useActivityStore, import.meta.hot));
}
