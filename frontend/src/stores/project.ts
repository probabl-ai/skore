import { defineStore } from "pinia";
import { ref, shallowRef } from "vue";

import { type Layout, type Project, type ProjectItem } from "@/models";
import { deleteView as deleteViewApi, fetchProject, putView } from "@/services/api";
import { poll } from "@/services/utils";

export interface TreeNode {
  name: string;
  children: TreeNode[];
}

export const useProjectStore = defineStore("project", () => {
  // this object is not deeply reactive as it may be very large
  const items = shallowRef<{ [key: string]: ProjectItem } | null>(null);
  const views = ref<{ [key: string]: Layout }>({});
  const currentView = ref<string | null>(null);

  /**
   * Return true if the the given key is in the list of displayed keys, false otherwise.
   * @param view the view to check
   * @param key the key to look for
   * @returns true if the key is displayed, false otherwise
   */
  function isKeyDisplayed(view: string, key: string) {
    const visibleKeys = views.value[view] ?? [];
    return visibleKeys.includes(key);
  }

  /**
   * Add the value of a key to the view.
   * @param view the view to add the key to
   * @param key the key to add to the view
   */
  async function displayKey(view: string, key: string) {
    stopBackendPolling();
    const realKey = key.replace(" (self)", "");
    if (!isKeyDisplayed(view, realKey)) {
      views.value[view] = [...views.value[view], realKey];
      await persistView(view, views.value[view]);
    }
    await startBackendPolling();
  }

  /**
   * Hide the value of a key from the view.
   * @param view the view to hide the key from
   * @param key the key to hide
   */
  async function hideKey(view: string, key: string) {
    stopBackendPolling();
    if (isKeyDisplayed(view, key)) {
      const v = views.value[view];
      views.value[view] = v.filter((k) => k !== key);
      await persistView(view, views.value[view]);
    }
    await startBackendPolling();
  }

  /**
   * Fetch all reports URI
   * and eventually the detail of the currently selected view
   */
  let _isCanceledCall = false;
  async function fetch() {
    if (!_isCanceledCall) {
      const r = await fetchProject();
      if (r) {
        setProject(r);
      }
    }
  }
  /**
   * Start real time sync with the server.
   */
  let _stopBackendPolling: Function | null = null;
  async function startBackendPolling() {
    _isCanceledCall = false;
    await fetch();
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
   * Send the view's layout to backend
   * @param view the view to persist
   * @param layout the layout to persist
   */
  async function persistView(view: string, layout: Layout) {
    if (items.value && views.value) {
      const r = await putView(view, layout);
      if (r) {
        setProject(r);
      }
    }
  }

  /**
   * Set the current view and populate the layout
   * @param r data received from the backend
   */
  async function setProject(r: Project) {
    items.value = r.items;
    views.value = r.views;
  }

  /**
   * Transform a list of leys into a tree of FileTreeNodes.
   *
   * i.e. `["a", "a/b", "a/b/d", "a/b/e", "a/b/f/g"]` ->
   * ```json
   * [
   *   {
   *     name: "a",
   *     children: [
   *       { name: "a (self)", children: [] },
   *       {
   *         name: "a/b",
   *         children: [
   *           { name: "a/b (self)", children: [] },
   *           { name: "a/b/d", children: [] },
   *           { name: "a/b/e", children: [] },
   *           {
   *             name: "a/b/f",
   *             children: [
   *               { name: "a/b/f/g", children: [] },
   *             ],
   *           },
   *         ],
   *       },
   *     ],
   *   },
   * ]
   * ```
   *
   * @param list - A list of strings to transform into a tree.
   * @returns A tree of FileTreeNodes.
   */
  function keysAsTree() {
    const lut: { [key: string]: TreeNode } = {};
    const tree: TreeNode[] = [];
    const keys = Object.keys(items.value || {});

    for (const key of keys) {
      const segments = key.split("/").filter((s) => s.length > 0);
      const rootSegment = segments[0];
      let currentNode = lut[rootSegment];
      if (!currentNode) {
        currentNode = { name: rootSegment, children: [] };
        tree.push(currentNode);
        lut[rootSegment] = currentNode;
      }
      let n = currentNode!;
      for (const s of segments.slice(1)) {
        n.children = n.children || [];
        const name = `${n.name}/${s}`;
        let childNode = lut[name];
        if (!childNode) {
          childNode = { name, children: [] };
          n.children.push(childNode);
          lut[name] = childNode;
        }
        n = childNode;
      }
    }

    // add (self) to the name of the node if it is the same as the name of the key
    function addSelf(node: TreeNode) {
      if (keys.includes(node.name) && node.children.length > 0) {
        node.children.unshift({ name: `${node.name} (self)`, children: [] });
      }
      for (const child of node.children) {
        addSelf(child);
      }
    }

    for (const node of tree) {
      addSelf(node);
    }
    return tree;
  }

  /**
   * Create a new view with the given name.
   * @param name the name of the view to create
   */
  async function createView(name: string) {
    views.value[name] = [];
    await persistView(name, views.value[name]);
  }

  /**
   * Duplicate a view.
   * @param src the source view to duplicate
   * @param name the name of the new view
   */
  async function duplicateView(src: string, name: string) {
    views.value[name] = [...views.value[src]];
    await persistView(name, views.value[name]);
  }

  /**
   * Delete a view.
   * @param name the name of the view to delete
   */
  async function deleteView(name: string) {
    if (name === currentView.value) {
      currentView.value = null;
    }
    delete views.value[name];
    await deleteViewApi(name);
  }

  /**
   * Rename a view.
   * @param name the name of the view to rename
   */
  async function renameView(name: string, newName: string) {
    views.value[newName] = views.value[name];
    delete views.value[name];
    await deleteView(name);
    await persistView(newName, views.value[newName]);
  }

  return {
    items,
    views,
    currentView,
    displayKey,
    hideKey,
    startBackendPolling,
    stopBackendPolling,
    setProject,
    keysAsTree,
    createView,
    duplicateView,
    deleteView,
    renameView,
  };
});
