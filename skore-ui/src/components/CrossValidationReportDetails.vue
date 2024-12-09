<script setup lang="ts">
import type { DetailSection } from "@/components/CrossValidationReport.vue";

const props = defineProps<{ sections: DetailSection[] }>();
</script>

<template>
  <div class="cross-validation-report-details">
    <div class="section" v-for="section in props.sections" :key="section.title">
      <div class="title">{{ section.title }}</div>
      <div class="items">
        <div class="item" v-for="item in section.items" :key="item.name">
          <div class="name-and-description">
            <div class="name">{{ item.name }}</div>
            <dvi class="description">{{ item.description }}</dvi>
          </div>
          <div class="value">{{ item.value }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style>
.cross-validation-report-details {
  padding: var(--spacing-16);

  & .section {
    & .title {
      color: var(--color-text-primary);
      font-size: var(--font-size-sm);
    }

    & .items {
      display: flex;
      flex-direction: column;
      font-size: var(--font-size-xs);
      gap: var(--spacing-20);

      & .item {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;

        & .name {
          color: var(--color-text-primary);
        }

        & .description {
          color: var(--color-text-secondary);
        }

        & .value {
          color: var(--color-text-branding);
          text-align: right;
        }
      }
    }
  }
}
</style>
