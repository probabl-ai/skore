import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";

import TreeAccordion, { type TreeAccordionNode } from "@/components/TreeAccordion.vue";

function countLeaves(nodes: TreeAccordionNode[]): number {
  function countInNode(node: TreeAccordionNode): number {
    if (node.children?.length === 0) {
      return 1;
    }

    const countInChildren = node.children?.map(countInNode) ?? [];
    return countInChildren.reduce((accumulator, leavesCount) => accumulator + leavesCount, 0);
  }

  const allBranches = nodes.map(countInNode);
  return allBranches.reduce((accumulator, leavesCount) => accumulator + leavesCount);
}

describe("TreeAccordion", () => {
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
    const records: TreeAccordionNode[] = [
      {
        name: "Punk Rock",
        enabled: true,
        children: [
          {
            name: "The Clash",
            enabled: true,
            children: [
              { name: "The Clash", enabled: true },
              { name: "Give 'Em Enough Rope", enabled: true },
              { name: "London Calling", enabled: true },
              { name: "Sandinista!", enabled: true },
              { name: "Combat Rock", enabled: true },
              { name: "Cut the Crap", enabled: true },
            ],
          },
          {
            name: "Ramones",
            enabled: true,
            children: [
              { name: "Ramones", enabled: true },
              { name: "Leave Home", enabled: true },
              { name: "Rocket to Russia", enabled: true },
              { name: "Road to Ruin", enabled: true },
              { name: "End of the Century", enabled: true },
              { name: "Pleasant Dreams", enabled: true },
              { name: "Subterranean Jungle", enabled: true },
              { name: "Too Tough to Die", enabled: true },
              { name: "Animal Boy", enabled: true },
              { name: "Halfway to Sanity", enabled: true },
              { name: "Brain Drain", enabled: true },
              { name: "Mondo Bizarro", enabled: true },
              { name: "Acid Eaters", enabled: true },
              { name: "¡Adios Amigos!", enabled: true },
            ],
          },
        ],
      },
      {
        name: "French touch",
        enabled: true,
        children: [
          {
            name: "Laurent Garnier",
            enabled: true,
            children: [
              { name: "Shot in the Dark", enabled: true },
              { name: "Club Traxx EP", enabled: true },
              { name: "30", enabled: true },
              { name: "Early Works", enabled: true },
              { name: "Unreasonable Behaviour", enabled: true },
              { name: "The Cloud Making Machine", enabled: true },
              { name: "Retrospective", enabled: true },
              { name: "Public Outburst", enabled: true },
              { name: "Tales of a Kleptomaniac", enabled: true },
              { name: "Suivront Mille Ans De Calme", enabled: true },
              { name: "Home Box", enabled: true },
              { name: "Paris Est à Nous", enabled: true },
              { name: "Le Roi Bâtard", enabled: true },
              { name: "De Película", enabled: true },
              { name: "Entre la Vie et la Mort", enabled: true },
              { name: "33 tours et puis s'en vont", enabled: true },
            ],
          },
        ],
      },
    ];

    const wrapper = mount(TreeAccordion, {
      props: { nodes: records },
    });

    const itemSelector = ".tree-accordion-item";
    const treeItems = wrapper.findAllComponents(itemSelector);
    const leavesCount = countLeaves(records);
    const leaves = treeItems.filter((c) => c.findAll(itemSelector).length == 0);
    expect(leaves).toHaveLength(leavesCount);
  });
});
