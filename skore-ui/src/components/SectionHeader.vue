<script setup lang="ts">
import FloatingTooltip from "@/components/FloatingTooltip.vue";
import SimpleButton from "@/components/SimpleButton.vue";

const props = defineProps<{
  title: string;
  subtitle?: string;
  icon?: string;
  actionIcon?: string;
  actionTooltip?: string;
}>();

const emit = defineEmits(["action"]);
</script>

<template>
  <div class="header">
    <h1>
      <span v-if="props.icon" class="icon" :class="props.icon"></span>{{ title }}
      <span v-if="props.subtitle" class="subtitle">
        <span class="separator">-</span>
        {{ props.subtitle }}
      </span>
    </h1>
    <div class="action" v-if="props.actionIcon">
      <FloatingTooltip :text="props.actionTooltip" v-if="props.actionTooltip">
        <SimpleButton :icon="props.actionIcon" @click="emit('action')" />
      </FloatingTooltip>
      <SimpleButton v-else :icon="props.actionIcon" @click="emit('action')" />
    </div>
  </div>
</template>

<style scoped>
.header {
  display: flex;
  height: 44px;
  align-items: center;
  justify-content: space-between;
  padding: var(--spacing-padding-large);
  border-right: solid var(--border-width-normal) var(--border-color-normal);
  border-bottom: solid var(--border-width-normal) var(--border-color-normal);
  background-color: var(--background-color-elevated);

  & h1 {
    color: var(--text-color-normal);
    font-size: var(--text-size-title);
    font-weight: var(--text-weight-title);
  }

  & .icon {
    padding-right: 4px;
  }

  & .subtitle {
    color: var(--text-color-low);
    font-size: var(--text-size-normal);
    font-weight: var(--text-weight-normal);

    & .separator {
      margin: 0 var(--spacing-gap-small);
    }
  }
}
</style>
