<script setup lang="ts">
import { RouterView, useRoute, useRouter } from "vue-router";

import LoadingBars from "@/components/LoadingBars.vue";
import ModalDialog from "@/components/ModalDialog.vue";
import NavigationButton from "@/components/NavigationButton.vue";
import ToastNotificationArea from "@/components/ToastNotificationArea.vue";
import { useThemesStore } from "@/stores/themes";
import AppToolbar from "@/views/AppToolbar.vue";

const route = useRoute();
const router = useRouter();
const themesStore = useThemesStore();
</script>

<template>
  <div class="skore" :class="themesStore.currentThemeClass">
    <div class="app-shell">
      <AppToolbar>
        <template v-if="router.getRoutes().length > 1">
          <NavigationButton
            v-for="(r, i) in router.getRoutes()"
            :key="i"
            :icon="`${r.meta['icon']}`"
            :is-selected="r.name == route.name"
            :to="r.path"
          />
        </template>
      </AppToolbar>
      <RouterView v-slot="{ Component, route }">
        <template v-if="Component">
          <Transition name="fade" mode="out-in">
            <Suspense timeout="0">
              <template #default>
                <component :is="Component" :key="route.path" />
              </template>
              <template #fallback>
                <div class="loader">
                  <LoadingBars />
                </div>
              </template>
            </Suspense>
          </Transition>
        </template>
      </RouterView>
    </div>
    <ToastNotificationArea />
    <ModalDialog />
  </div>
</template>

<style scoped>
.skore {
  color: var(--color-text-primary);

  & .app-shell {
    display: flex;
    height: 100dvh;
    flex-direction: column;
    background-color: var(--color-background-primary);

    main,
    .loader {
      flex: 1;
    }

    .loader {
      display: flex;
      align-items: center;
      justify-content: center;
    }
  }
}
</style>
