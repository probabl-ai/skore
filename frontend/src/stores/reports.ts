import { defineStore } from "pinia";
import { ref, shallowRef, toRaw } from "vue";

import { DataStore, type KeyLayoutSize, type KeyMoveDirection, type Layout } from "@/models";
import { fetchAllManderUris, fetchMander, putLayout } from "@/services/api";
import { isDeepEqual, poll, swapItemsInArray } from "@/services/utils";

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
    stopBackendPolling();
    if (!isKeyDisplayed(key)) {
      layout.value.push({ key, size: "large" });
      await persistLayout();
    }
    await startBackendPolling();
  }

  /**
   * Hide the value of a key from the report.
   * @param key the key to hide
   */
  async function hideKey(key: string) {
    stopBackendPolling();
    if (isKeyDisplayed(key)) {
      layout.value = layout.value.filter(({ key: k }) => key != k);
      await persistLayout();
    }
    await startBackendPolling();
  }

  /**
   * Change the visual size of the value of a key in the report.
   * @param key the key which changes
   * @param size the target size
   */
  async function setKeyLayoutSize(key: string, size: KeyLayoutSize) {
    stopBackendPolling();
    if (isKeyDisplayed(key)) {
      const index = layout.value.findIndex(({ key: k }) => key == k);
      if (index !== -1) {
        layout.value[index].size = size;
      }
      await persistLayout();
    }
    await startBackendPolling();
  }

  /**
   * Move a displayed key up or down in the list
   * @param key the key to replace
   * @param direction up or down
   */
  async function moveKey(key: string, direction: KeyMoveDirection) {
    stopBackendPolling();
    const offset = direction == "up" ? -1 : 1;
    const index = layout.value.findIndex(({ key: k }) => key == k);
    swapItemsInArray(layout.value, index, index + offset);
    await persistLayout();
    await startBackendPolling();
  }

  /**
   * Set the selected report ref to a new one only if there is a difference.
   * @param report the new report
   */
  function setSelectedReportIfDifferent(report: DataStore) {
    const current = toRaw(selectedReport.value);
    if (!isDeepEqual(current as object, report)) {
      selectedReport.value = report;
      layout.value = report.layout;
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
          setSelectedReportIfDifferent(report);
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
    _stopBackendPolling = await poll(fetch, 1500);
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
    if (selectedReport.value && layout.value) {
      const report = await putLayout(selectedReport.value.uri, layout.value);
      if (report) {
        setSelectedReportIfDifferent(report);
      }
    }
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
    setSelectedReportIfDifferent,
    moveKey,
  };
});
