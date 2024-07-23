import { describe, expect, it } from "vitest";

import FileTree, { transformUrisToTree, type FileTreeNode } from "@/components/FileTree.vue";
import { mount } from "@vue/test-utils";

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
  it("Renders properly.", () => {
    const records: FileTreeNode[] = [
      {
        uri: "Punk Rock",
        children: [
          {
            uri: "The Clash",
            children: [
              { uri: "The Clash" },
              { uri: "Give 'Em Enough Rope" },
              { uri: "London Calling" },
              { uri: "Sandinista!" },
              { uri: "Combat Rock" },
              { uri: "Cut the Crap" },
            ],
          },
          {
            uri: "Ramones",
            children: [
              { uri: "Ramones" },
              { uri: "Leave Home" },
              { uri: "Rocket to Russia" },
              { uri: "Road to Ruin" },
              { uri: "End of the Century" },
              { uri: "Pleasant Dreams" },
              { uri: "Subterranean Jungle" },
              { uri: "Too Tough to Die" },
              { uri: "Animal Boy" },
              { uri: "Halfway to Sanity" },
              { uri: "Brain Drain" },
              { uri: "Mondo Bizarro" },
              { uri: "Acid Eaters" },
              { uri: "¡Adios Amigos!" },
            ],
          },
        ],
      },
      {
        uri: "French touch",
        children: [
          {
            uri: "Laurent Garnier",
            children: [
              { uri: "Shot in the Dark" },
              { uri: "Club Traxx EP" },
              { uri: "30" },
              { uri: "Early Works" },
              { uri: "Unreasonable Behaviour" },
              { uri: "The Cloud Making Machine" },
              { uri: "Retrospective" },
              { uri: "Public Outburst" },
              { uri: "Tales of a Kleptomaniac" },
              { uri: "Suivront Mille Ans De Calme" },
              { uri: "Home Box" },
              { uri: "Paris Est à Nous" },
              { uri: "Le Roi Bâtard" },
              { uri: "De Película" },
              { uri: "Entre la Vie et la Mort" },
              { uri: "33 tours et puis s'en vont" },
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

    const tree = transformUrisToTree(uris);
    expect(tree).toEqual([
      {
        uri: "probabl-ai",
        children: [
          {
            uri: "probabl-ai/demo-usecase",
            children: [
              {
                uri: "probabl-ai/demo-usecase/training",
                children: [{ uri: "probabl-ai/demo-usecase/training/0" }],
              },
            ],
          },
          {
            uri: "probabl-ai/test-mandr",
            children: [
              { uri: "probabl-ai/test-mandr/0" },
              { uri: "probabl-ai/test-mandr/1" },
              { uri: "probabl-ai/test-mandr/2" },
              { uri: "probabl-ai/test-mandr/3" },
              { uri: "probabl-ai/test-mandr/4" },
            ],
          },
        ],
      },
    ]);
  });
});
