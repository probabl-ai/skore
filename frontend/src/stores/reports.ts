import { debounce } from "lodash";
import { defineStore } from "pinia";
import { ref, shallowRef, toRaw } from "vue";

import { DataStore, type KeyLayoutSize, type Layout } from "@/models";
import { fetchAllManderUris, fetchMander, putLayout } from "@/services/api";
import { poll } from "@/services/utils";

export const useReportsStore = defineStore("reports", () => {
  // this object is not deeply reactive as it may be very large
  const reportUris = ref<string[]>([]);
  const selectedReportUri = ref<string>("");
  const report = shallowRef<DataStore | null>(null);
  const layout = ref<Layout>([]);

  async function displayKey(key: string) {
    layout.value.push({ key, size: "large" });
    await persistLayout();
  }

  async function hideKey(key: string) {
    layout.value = layout.value.filter(({ key: k }) => key != k);
    await persistLayout();
  }

  async function setKeyLayoutSize(key: string, size: KeyLayoutSize) {
    const index = layout.value.findIndex(({ key: k }) => key == k);
    if (index !== -1) {
      layout.value[index].size = size;
    }
    await persistLayout();
  }

  /**
   * merge report
   * thruth comes from the backend except card size that the user may have change
   */
  function merge() {
    const r = report.value;
    if (r) {
      // merge layout
      const old = structuredClone(toRaw(layout.value));
      const allKeys = Object.keys(r.payload);
      const displayedKeys = layout.value.map((v) => v.key).filter((v) => allKeys.indexOf(v) !== -1);
      const newLayout = [];
      for (const { key, size: s } of r.layout) {
        if (displayedKeys.indexOf(key) !== -1) {
          const localSizeIndex = old.findIndex(({ key: k }) => key == k);
          const size = localSizeIndex !== -1 ? old[localSizeIndex].size : s;
          newLayout.push({ key, size });
        }
      }
      layout.value = newLayout;
    }
  }

  /**
   * Fetch all reports URI
   * and eventually the detail of the currently selected report
   */
  async function fetch() {
    reportUris.value = await fetchAllManderUris();
    const selectedUri = selectedReportUri.value;
    if (selectedUri.length > 0) {
      report.value = await fetchMander(selectedUri);
      merge();
    }
  }

  /**
   * Start real time sync with the server.
   */
  let _stopBackendPolling: Function | null = null;
  async function startBackendPolling() {
    _stopBackendPolling = await poll(fetch, 500);
  }

  /**
   * Stop real time sync with the server.
   */
  function stopBackendPolling() {
    _stopBackendPolling && _stopBackendPolling();
  }

  /**
   * Send new layout to backend
   */
  async function _persistLayout() {
    stopBackendPolling();
    if (report.value && layout.value) {
      const refreshed = await putLayout(report.value.uri, layout.value);
      if (refreshed) {
        report.value = refreshed;
      }
    }
    await startBackendPolling();
  }
  /**
   * Debounced layout sync with the backend.
   * To avoid server spamming.
   */
  const persistLayout = debounce(_persistLayout, 1000, { leading: true, trailing: false });

  return {
    reportUris,
    selectedReportUri,
    report,
    layout,
    displayKey,
    hideKey,
    setKeyLayoutSize,
    startBackendPolling,
    stopBackendPolling,
  };
});
