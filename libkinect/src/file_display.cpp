/*
   Novelty face authentication with liveness detection using depth and IR camera
   Copyright (C) 2017-2018
   Tomasz Garbus, Dominik Klemba, Jan Ludziejewski, Łukasz Raszkiewicz

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <string>

#include <wx/bitmap.h>
#include <wx/event.h>
#include <wx/image.h>
#include <wx/panel.h>
#include <wx/slider.h>
#include <wx/stattext.h>
#include <wx/wx.h>
#include <wx/wxprec.h>

#include "picture.hpp"

// Constants

enum { ID_MIN_D = 101, ID_MAX_D = 102, ID_MIN_D_TEXT = 103, ID_MAX_D_TEXT = 104, ID_DISPLAY = 105 };

wxDEFINE_EVENT(REFRESH_DISPLAY_EVENT, wxCommandEvent);

// Declarations

class MainWindow;

class DisplayPanel : public wxPanel {
   wxStaticBitmap *m_picture = nullptr;

 public:
   DisplayPanel(wxPanel *parent, wxWindowID window_id, MainWindow *window, Picture::DepthOrIrFrame *frame);

   void refresh_display(wxCommandEvent &event);

   MainWindow *window;
   Picture::DepthOrIrFrame *frame;
   uint8_t *bitmap;
};

class SettingsPanel : public wxPanel {
 public:
   explicit SettingsPanel(
         wxPanel *parent, MainWindow *window, int min_slider_default, int max_slider_default, int slider_max);

   void on_min_slider_change(wxCommandEvent &event);
   void on_max_slider_change(wxCommandEvent &event);
   void on_min_text_change(wxCommandEvent &event);
   void on_max_text_change(wxCommandEvent &event);

   MainWindow *window;
   wxPanel *m_parent;
   wxSlider *m_min_d, *m_max_d;
   wxTextCtrl *m_min_d_text, *m_max_d_text;
};

class MainWindow : public wxFrame {
   wxPanel *m_parent;

 public:
   explicit MainWindow(const wxString &title, Picture::DepthOrIrFrame *frame);

   DisplayPanel *m_display;
   SettingsPanel *m_settings;
};

// Definitions

SettingsPanel::SettingsPanel(
      wxPanel *parent, MainWindow *window, int min_slider_default, int max_slider_default, int slider_max)
      : wxPanel(parent, -1, wxPoint(-1, -1), wxSize(-1, -1), wxBORDER_SUNKEN), m_parent(parent), window(window),
        m_min_d(new wxSlider(this, ID_MIN_D, min_slider_default, 0, slider_max, wxPoint(80, 10), wxSize(500, 15))),
        m_max_d(new wxSlider(this, ID_MAX_D, max_slider_default, 0, slider_max, wxPoint(80, 40), wxSize(500, 15))),
        m_min_d_text(
              new wxTextCtrl(this, ID_MIN_D_TEXT, std::to_string(min_slider_default), wxPoint(10, 10), wxSize(60, 15))),
        m_max_d_text(new wxTextCtrl(
              this, ID_MAX_D_TEXT, std::to_string(max_slider_default), wxPoint(10, 40), wxSize(60, 15))) {
   m_min_d->Bind(wxEVT_SCROLL_CHANGED, &SettingsPanel::on_min_slider_change, this);
   m_min_d->Bind(wxEVT_SCROLL_THUMBTRACK, &SettingsPanel::on_min_slider_change, this);
   m_max_d->Bind(wxEVT_SCROLL_CHANGED, &SettingsPanel::on_max_slider_change, this);
   m_max_d->Bind(wxEVT_SCROLL_THUMBTRACK, &SettingsPanel::on_max_slider_change, this);
   m_min_d_text->Bind(wxEVT_TEXT, &SettingsPanel::on_min_text_change, this);
   m_max_d_text->Bind(wxEVT_TEXT, &SettingsPanel::on_max_text_change, this);
}

void SettingsPanel::on_min_slider_change(wxCommandEvent &event) {
   if (m_max_d->GetValue() <= m_min_d->GetValue()) {
      m_max_d->SetValue(m_min_d->GetValue() + 1);
      m_max_d_text->Clear();
      m_max_d_text->WriteText(std::to_string(m_min_d->GetValue() + 1));
   }

   m_min_d_text->Clear();
   m_min_d_text->WriteText(std::to_string(m_min_d->GetValue()));
   wxPostEvent(window->m_display, wxCommandEvent(REFRESH_DISPLAY_EVENT));
}

void SettingsPanel::on_max_slider_change(wxCommandEvent &event) {
   if (m_max_d->GetValue() <= m_min_d->GetValue()) {
      m_min_d->SetValue(m_max_d->GetValue() - 1);
      m_min_d_text->Clear();
      m_min_d_text->WriteText(std::to_string(m_max_d->GetValue() - 1));
   }

   m_max_d_text->Clear();
   m_max_d_text->WriteText(std::to_string(m_max_d->GetValue()));
   wxPostEvent(window->m_display, wxCommandEvent(REFRESH_DISPLAY_EVENT));
}

void SettingsPanel::on_min_text_change(wxCommandEvent &event) {
   long min_d;
   if (!m_min_d_text->GetValue().ToLong(&min_d)) {
      return;
   }
   m_min_d->SetValue(static_cast<int>(min_d));

   if (m_max_d->GetValue() <= min_d) {
      m_max_d->SetValue(static_cast<int>(min_d) + 1);
      m_max_d_text->Clear();
      m_max_d_text->WriteText(std::to_string(min_d + 1));
   }
   wxPostEvent(window->m_display, wxCommandEvent(REFRESH_DISPLAY_EVENT));
}

void SettingsPanel::on_max_text_change(wxCommandEvent &event) {
   long max_d;
   if (!m_max_d_text->GetValue().ToLong(&max_d)) {
      return;
   }
   m_max_d->SetValue(static_cast<int>(max_d));

   if (max_d <= m_min_d->GetValue()) {
      m_min_d->SetValue(static_cast<int>(max_d) - 1);
      m_min_d_text->Clear();
      m_min_d_text->WriteText(std::to_string(max_d - 1));
   }
   wxPostEvent(window->m_display, wxCommandEvent(REFRESH_DISPLAY_EVENT));
}

DisplayPanel::DisplayPanel(wxPanel *parent, wxWindowID window_id, MainWindow *window, Picture::DepthOrIrFrame *frame)
      : wxPanel(parent, window_id, wxPoint(0, 0),
              wxSize(static_cast<int>(frame->pixels->width), static_cast<int>(frame->pixels->height)), wxBORDER_SUNKEN),
        frame(frame), bitmap(new uint8_t[frame->pixels->width * frame->pixels->height * 3]), window(window) {
   Bind(REFRESH_DISPLAY_EVENT, &DisplayPanel::refresh_display, this);
   wxPostEvent(this, wxCommandEvent(REFRESH_DISPLAY_EVENT));
}

void DisplayPanel::refresh_display(wxCommandEvent &event) {
   delete m_picture;
   float min_depth = window->m_settings->m_min_d->GetValue();
   float max_depth = window->m_settings->m_max_d->GetValue();
   if (max_depth - min_depth < 1.0) {
      max_depth = min_depth + 1.0f;
   }
   size_t width = frame->pixels->width;
   size_t height = frame->pixels->height;
   auto int_pixels = new uint8_t[width * height];
   for (size_t i = 0; i < width * height; ++i) {
      float val = 255.0f * (frame->pixels->data()[i] - min_depth) / (max_depth - min_depth);
      val = std::min(val, 255.0f);
      val = std::max(val, 0.0f);
      int_pixels[i] = static_cast<uint8_t>(val);
      if (!frame->is_depth) {
         bitmap[3 * i] = int_pixels[i];
         bitmap[3 * i + 1] = int_pixels[i];
         bitmap[3 * i + 2] = int_pixels[i];
      }
   }
   if (frame->is_depth) {
      cv::Mat current_image(cv::Size(static_cast<int>(width), static_cast<int>(height)), CV_8UC1, int_pixels);
      cv::Mat destination_image(cv::Size(static_cast<int>(width), static_cast<int>(height)), CV_8UC3);
      cv::applyColorMap(current_image, destination_image, cv::COLORMAP_RAINBOW);
      for (size_t i = 0; i < height; ++i) {
         for (size_t j = 0; j < width; ++j) {
            auto pixel = destination_image.at<cv::Vec3b>(static_cast<int>(i), static_cast<int>(j));
            bitmap[3 * (i * width + j)] = pixel[2];
            bitmap[3 * (i * width + j) + 1] = pixel[1];
            bitmap[3 * (i * width + j) + 2] = pixel[0];
         }
      }
   }
   m_picture = new wxStaticBitmap(this, wxID_ANY,
         wxBitmap(wxImage(static_cast<int>(width), static_cast<int>(height), bitmap, true)), wxDefaultPosition,
         wxDefaultSize);
}

MainWindow::MainWindow(const wxString &title, Picture::DepthOrIrFrame *frame)
      : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition,
              wxSize(static_cast<int>(frame->pixels->width + 150), static_cast<int>(frame->pixels->height + 200))),
        m_parent(new wxPanel(this, wxID_ANY)), m_display(new DisplayPanel(m_parent, ID_DISPLAY, this, frame)) {
   int min_slider_default, max_slider_default, slider_max;
   float max_value = 0.0;
   for (size_t i = 0; i < frame->pixels->width * frame->pixels->height; ++i) {
      max_value = std::max(max_value, frame->pixels->data()[i]);
   }
   if (frame->is_depth) {
      min_slider_default = 500;
      slider_max = static_cast<int>(max_value);
      max_slider_default = std::min(1500, slider_max);
   } else {
      min_slider_default = 0;
      max_slider_default = static_cast<int>(max_value);
      slider_max = max_slider_default;
   }
   m_settings = new SettingsPanel(m_parent, this, min_slider_default, max_slider_default, slider_max);
   auto vbox = new wxBoxSizer(wxVERTICAL);
   vbox->Add(m_display, 1, wxEXPAND | wxALL, 5);
   vbox->Add(m_settings, 1, wxEXPAND | wxALL, 5);
   m_parent->SetSizer(vbox);
   Centre();
}

// Main

class AppMain : public wxApp {
 public:
   bool OnInit() override;
   Picture::DepthOrIrFrame *frame = nullptr;
};

bool AppMain::OnInit() {
   MainWindow *window = new MainWindow(wxT("Depth/IR file display"), frame);
   window->Show(true);
   return true;
}

int main(int argc, char **argv) {
   auto app = new AppMain();
   app->frame = new Picture::DepthOrIrFrame(argv[1]);
   wxApp::SetInstance(app);
   return wxEntry(argc, argv);
}
