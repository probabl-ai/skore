import { defineStore } from "pinia";
import { ref, shallowRef } from "vue";

import { DataStore, type KeyLayoutSize, type Layout } from "@/models";
import { fetchAllManderUris, fetchMander, putLayout } from "@/services/api";
import { poll } from "@/services/utils";

export const useReportsStore = defineStore("reports", () => {
  // this object is not deeply reactive as it may be very large
  const reportUris = ref<string[]>([]);
  const selectedReportUri = ref<string>("");
  const selectedReport = shallowRef<DataStore | null>(null);
  const layout = ref<Layout>([]);

  /**
   * Return true if the the given key is in the list of displayed keys, false otherwise.
   * @param key the key to look for
   */
  function isKeyDisplayed(key: string) {
    const visibleKeys = layout.value.map(({ key: k }) => k);
    return visibleKeys.includes(key);
  }

  /**
   * Add the value of a key to the report.
   * @param key the key to add to the report
   */
  async function displayKey(key: string) {
    if (!isKeyDisplayed(key)) {
      layout.value.push({ key, size: "large" });
      await persistLayout();
    }
  }

  /**
   * Hide the value of a key from the report.
   * @param key the key to hide
   */
  async function hideKey(key: string) {
    if (isKeyDisplayed(key)) {
      layout.value = layout.value.filter(({ key: k }) => key != k);
      await persistLayout();
    }
  }

  /**
   * Change the visual size of the value of a key in the report.
   * @param key the key which changes
   * @param size the target size
   */
  async function setKeyLayoutSize(key: string, size: KeyLayoutSize) {
    if (isKeyDisplayed(key)) {
      const index = layout.value.findIndex(({ key: k }) => key == k);
      if (index !== -1) {
        layout.value[index].size = size;
      }
      await persistLayout();
    }
  }

  /**
   * Fetch all reports URI
   * and eventually the detail of the currently selected report
   */
  let _isCanceledCall = false;
  async function fetch() {
    reportUris.value = await fetchAllManderUris();
    if (!_isCanceledCall) {
      const selectedUri = selectedReportUri.value;
      if (selectedUri.length > 0) {
        const report = await fetchMander(selectedUri);
        if (report) {
          selectedReport.value = report;
          layout.value = report.layout;
        }
      }
    }
  }

  /**
   * Start real time sync with the server.
   */
  let _stopBackendPolling: Function | null = null;
  async function startBackendPolling() {
    _isCanceledCall = false;
    _stopBackendPolling = await poll(fetch, 500);
  }

  /**
   * Stop real time sync with the server.
   */
  function stopBackendPolling() {
    _isCanceledCall = true;
    _stopBackendPolling && _stopBackendPolling();
  }

  /**
   * Send new layout to backend
   */
  async function persistLayout() {
    stopBackendPolling();
    if (selectedReport.value && layout.value) {
      const report = await putLayout(selectedReport.value.uri, layout.value);
      if (report) {
        selectedReport.value = report;
      }
    }
    await startBackendPolling();
  }

  return {
    reportUris,
    selectedReportUri,
    selectedReport,
    layout,
    displayKey,
    hideKey,
    setKeyLayoutSize,
    startBackendPolling,
    stopBackendPolling,
  };
});
