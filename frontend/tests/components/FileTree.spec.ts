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
        path: "Punk Rock",
        children: [
          {
            path: "The Clash",
            children: [
              { path: "The Clash" },
              { path: "Give 'Em Enough Rope" },
              { path: "London Calling" },
              { path: "Sandinista!" },
              { path: "Combat Rock" },
              { path: "Cut the Crap" },
            ],
          },
          {
            path: "Ramones",
            children: [
              { path: "Ramones" },
              { path: "Leave Home" },
              { path: "Rocket to Russia" },
              { path: "Road to Ruin" },
              { path: "End of the Century" },
              { path: "Pleasant Dreams" },
              { path: "Subterranean Jungle" },
              { path: "Too Tough to Die" },
              { path: "Animal Boy" },
              { path: "Halfway to Sanity" },
              { path: "Brain Drain" },
              { path: "Mondo Bizarro" },
              { path: "Acid Eaters" },
              { path: "¡Adios Amigos!" },
            ],
          },
        ],
      },
      {
        path: "French touch",
        children: [
          {
            path: "Laurent Garnier",
            children: [
              { path: "Shot in the Dark" },
              { path: "Club Traxx EP" },
              { path: "30" },
              { path: "Early Works" },
              { path: "Unreasonable Behaviour" },
              { path: "The Cloud Making Machine" },
              { path: "Retrospective" },
              { path: "Public Outburst" },
              { path: "Tales of a Kleptomaniac" },
              { path: "Suivront Mille Ans De Calme" },
              { path: "Home Box" },
              { path: "Paris Est à Nous" },
              { path: "Le Roi Bâtard" },
              { path: "De Película" },
              { path: "Entre la Vie et la Mort" },
              { path: "33 tours et puis s'en vont" },
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
