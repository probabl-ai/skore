import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import FileTree, { type FileTreeNode, transformListToTree } from "@/components/FileTree.vue";
import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createApp } from "vue";

function countLeaves(nodes: FileTreeNode[]): number {
  function countInNode(node: FileTreeNode): number {
    if (!node.children?.length) {
      return 1;
    }

    const countInChildren = node.children.map(countInNode);
    return countInChildren.reduce((accumulator, leavesCount) => accumulator + leavesCount);
  }

  const allBranches = nodes.map(countInNode);
  return allBranches.reduce((accumulator, leavesCount) => accumulator + leavesCount);
}

describe("FileTree", () => {
  beforeEach(() => {
    vi.mock("vue-router");
    const app = createApp({});
    const pinia = createPinia();
    app.use(pinia);
    setActivePinia(pinia);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("Renders properly.", () => {
    const records: FileTreeNode[] = [
      {
        stem: "Punk Rock",
        children: [
          {
            stem: "The Clash",
            children: [
              { stem: "The Clash" },
              { stem: "Give 'Em Enough Rope" },
              { stem: "London Calling" },
              { stem: "Sandinista!" },
              { stem: "Combat Rock" },
              { stem: "Cut the Crap" },
            ],
          },
          {
            stem: "Ramones",
            children: [
              { stem: "Ramones" },
              { stem: "Leave Home" },
              { stem: "Rocket to Russia" },
              { stem: "Road to Ruin" },
              { stem: "End of the Century" },
              { stem: "Pleasant Dreams" },
              { stem: "Subterranean Jungle" },
              { stem: "Too Tough to Die" },
              { stem: "Animal Boy" },
              { stem: "Halfway to Sanity" },
              { stem: "Brain Drain" },
              { stem: "Mondo Bizarro" },
              { stem: "Acid Eaters" },
              { stem: "¡Adios Amigos!" },
            ],
          },
        ],
      },
      {
        stem: "French touch",
        children: [
          {
            stem: "Laurent Garnier",
            children: [
              { stem: "Shot in the Dark" },
              { stem: "Club Traxx EP" },
              { stem: "30" },
              { stem: "Early Works" },
              { stem: "Unreasonable Behaviour" },
              { stem: "The Cloud Making Machine" },
              { stem: "Retrospective" },
              { stem: "Public Outburst" },
              { stem: "Tales of a Kleptomaniac" },
              { stem: "Suivront Mille Ans De Calme" },
              { stem: "Home Box" },
              { stem: "Paris Est à Nous" },
              { stem: "Le Roi Bâtard" },
              { stem: "De Película" },
              { stem: "Entre la Vie et la Mort" },
              { stem: "33 tours et puis s'en vont" },
            ],
          },
        ],
      },
    ];

    const wrapper = mount(FileTree, {
      props: { nodes: records },
    });

    const itemSelector = ".file-tree-item";
    const treeItems = wrapper.findAll(itemSelector);
    const leavesCount = countLeaves(records);
    const leaves = treeItems.filter((c) => c.findAll(itemSelector).length == 0);
    expect(leaves).toHaveLength(leavesCount);
  });

  it("Can transform an array of URIs to a tree", () => {
    const uris = [
      "probabl-ai/demo-usecase/training/0",
      "probabl-ai/test-mandr/0",
      "probabl-ai/test-mandr/1",
      "probabl-ai/test-mandr/2",
      "probabl-ai/test-mandr/3",
      "probabl-ai/test-mandr/4",
    ];

    const tree = transformListToTree(uris);
    expect(tree).toEqual([
      {
        stem: "probabl-ai",
        children: [
          {
            stem: "probabl-ai/demo-usecase",
            children: [
              {
                stem: "probabl-ai/demo-usecase/training",
                children: [{ stem: "probabl-ai/demo-usecase/training/0" }],
              },
            ],
          },
          {
            stem: "probabl-ai/test-mandr",
            children: [
              { stem: "probabl-ai/test-mandr/0" },
              { stem: "probabl-ai/test-mandr/1" },
              { stem: "probabl-ai/test-mandr/2" },
              { stem: "probabl-ai/test-mandr/3" },
              { stem: "probabl-ai/test-mandr/4" },
            ],
          },
        ],
      },
    ]);
  });
});
