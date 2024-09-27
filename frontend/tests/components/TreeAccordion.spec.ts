import { mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createApp } from "vue";

import TreeAccordion, { type TreeAccordionNode } from "@/components/TreeAccordion.vue";

function countLeaves(nodes: TreeAccordionNode[]): number {
  function countInNode(node: TreeAccordionNode): number {
    if (!node.children?.length) {
      return 1;
    }

    const countInChildren = node.children.map(countInNode);
    return countInChildren.reduce((accumulator, leavesCount) => accumulator + leavesCount);
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
        children: [
          {
            name: "The Clash",
            children: [
              { name: "The Clash" },
              { name: "Give 'Em Enough Rope" },
              { name: "London Calling" },
              { name: "Sandinista!" },
              { name: "Combat Rock" },
              { name: "Cut the Crap" },
            ],
          },
          {
            name: "Ramones",
            children: [
              { name: "Ramones" },
              { name: "Leave Home" },
              { name: "Rocket to Russia" },
              { name: "Road to Ruin" },
              { name: "End of the Century" },
              { name: "Pleasant Dreams" },
              { name: "Subterranean Jungle" },
              { name: "Too Tough to Die" },
              { name: "Animal Boy" },
              { name: "Halfway to Sanity" },
              { name: "Brain Drain" },
              { name: "Mondo Bizarro" },
              { name: "Acid Eaters" },
              { name: "¡Adios Amigos!" },
            ],
          },
        ],
      },
      {
        name: "French touch",
        children: [
          {
            name: "Laurent Garnier",
            children: [
              { name: "Shot in the Dark" },
              { name: "Club Traxx EP" },
              { name: "30" },
              { name: "Early Works" },
              { name: "Unreasonable Behaviour" },
              { name: "The Cloud Making Machine" },
              { name: "Retrospective" },
              { name: "Public Outburst" },
              { name: "Tales of a Kleptomaniac" },
              { name: "Suivront Mille Ans De Calme" },
              { name: "Home Box" },
              { name: "Paris Est à Nous" },
              { name: "Le Roi Bâtard" },
              { name: "De Película" },
              { name: "Entre la Vie et la Mort" },
              { name: "33 tours et puis s'en vont" },
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
