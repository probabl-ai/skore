<script setup lang="ts">
import type { DetailSectionDto } from "@/dto";
import MarkdownIt from "markdown-it";

const props = defineProps<{ sections: DetailSectionDto[] }>();
const renderer = MarkdownIt({ html: true });

function itemAsHtml(v: string) {
  return renderer.render(v);
}
</script>

<template>
  <div class="cross-validation-report-details">
    <div class="section" v-for="section in props.sections" :key="section.title">
      <div class="title"><i class="icon" :class="section.icon" />{{ section.title }}</div>
      <div class="items">
        <div class="item" v-for="item in section.items" :key="item.name">
          <div class="name-and-description">
            <div class="name">{{ item.name }}</div>
            <dvi class="description">{{ item.description }}</dvi>
          </div>
          <div class="value" v-html="itemAsHtml(item.value)" />
        </div>
      </div>
    </div>
  </div>
</template>

<style>
.cross-validation-report-details {
  padding: var(--spacing-16);

  & .section {
    display: flex;
    flex-direction: column;
    padding-bottom: var(--spacing-24);
    gap: var(--spacing-16);

    & .title {
      margin-bottom: var(--spacing-4);
      color: var(--color-text-primary);
      font-size: var(--font-size-sm);

      & .icon {
        padding-right: var(--spacing-4);
        color: var(--color-text-branding);
      }
    }

    & .items {
      display: flex;
      flex-direction: column;
      font-size: var(--font-size-xs);
      gap: var(--spacing-20);

      & .item {
        display: flex;
        flex-direction: row;
        justify-content: space-between;

        & .name {
          color: var(--color-text-primary);
        }

        & .description {
          color: var(--color-text-secondary);
        }

        & .value {
          color: var(--color-text-secondary);
          font-family: GeistMono, monospace;

          & em,
          & code {
            color: var(--color-text-branding);
            font-style: normal;
            font-weight: var(--font-weight-medium);
          }

          & code {
            display: inline-block;
            padding: var(--spacing-4);
            border-radius: var(--radius-xs);
            background-color: rgb(from var(--color-text-branding) r g b / 20%);
          }

          & ul {
            padding-left: 15px;
          }
        }
      }
    }
  }
}
</style>
