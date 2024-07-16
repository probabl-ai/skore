import { describe, it, expect } from "vitest";

import { mount } from "@vue/test-utils";
import FileTree, { type FileTreeNode } from "@/components/FileTree.vue";

function countLeaves(nodes: FileTreeNode[]): number {
  function countInNode(node: FileTreeNode): number {
    if (!node.children?.length) {
      return 1;
    }

    const countInChildren = node.children.map((node) => countInNode(node));
    return countInChildren.reduce((accumulator, leavesCount) => accumulator + leavesCount);
  }

  const allBranches = nodes.map((node) => countInNode(node));
  return allBranches.reduce((accumulator, leavesCount) => accumulator + leavesCount);
}

describe("FileTree", () => {
  it("Renders properly.", () => {
    const records: FileTreeNode[] = [
      {
        label: "Punk Rock",
        children: [
          {
            label: "The Clash",
            children: [
              { label: "The Clash" },
              { label: "Give 'Em Enough Rope" },
              { label: "London Calling" },
              { label: "Sandinista!" },
              { label: "Combat Rock" },
              { label: "Cut the Crap" },
            ],
          },
          {
            label: "Ramones",
            children: [
              { label: "Ramones" },
              { label: "Leave Home" },
              { label: "Rocket to Russia" },
              { label: "Road to Ruin" },
              { label: "End of the Century" },
              { label: "Pleasant Dreams" },
              { label: "Subterranean Jungle" },
              { label: "Too Tough to Die" },
              { label: "Animal Boy" },
              { label: "Halfway to Sanity" },
              { label: "Brain Drain" },
              { label: "Mondo Bizarro" },
              { label: "Acid Eaters" },
              { label: "¡Adios Amigos!" },
            ],
          },
        ],
      },
      {
        label: "French touch",
        children: [
          {
            label: "Laurent Garnier",
            children: [
              { label: "Shot in the Dark" },
              { label: "Club Traxx EP" },
              { label: "30" },
              { label: "Early Works" },
              { label: "Unreasonable Behaviour" },
              { label: "The Cloud Making Machine" },
              { label: "Retrospective" },
              { label: "Public Outburst" },
              { label: "Tales of a Kleptomaniac" },
              { label: "Suivront Mille Ans De Calme" },
              { label: "Home Box" },
              { label: "Paris Est à Nous" },
              { label: "Le Roi Bâtard" },
              { label: "De Película" },
              { label: "Entre la Vie et la Mort" },
              { label: "33 tours et puis s'en vont" },
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
});
