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
#include "libkinect.hpp"

enum { ID_MIN_D = 101, ID_MAX_D = 102, ID_MIN_D_TEXT = 103, ID_MAX_D_TEXT = 104 };

// const int64_t display_width = 480;

wxDEFINE_EVENT(REFRESH_PICTURE_EVENT, wxCommandEvent);

class DisplayPanel;

class SettingsPanel : public wxPanel {
   DisplayPanel *picture_panel;

   wxPanel *m_parent;
   wxSlider *m_min_d, *m_max_d;
   wxTextCtrl *m_min_d_text, *m_max_d_text;

 public:
   SettingsPanel(wxPanel *parent, DisplayPanel *picture_panel);

   void on_change(wxCommandEvent &event);
};

class DisplayPanel : public wxPanel {
   Picture *const picture;
   wxStaticBitmap *m_picture = nullptr;

 public:
   DisplayPanel(wxPanel *parent, Picture *picture);

   void refresh_picture(wxCommandEvent &event);
};

class MainWindow : public wxFrame {
   Picture *picture;
   wxPanel *m_parent;

   SettingsPanel *m_settings;

 public:
   DisplayPanel *m_display;
   explicit MainWindow(const wxString &title);
};

// Definitions

SettingsPanel::SettingsPanel(wxPanel *parent, DisplayPanel *picture_panel)
      : wxPanel(parent, -1, wxPoint(-1, -1), wxSize(-1, -1), wxBORDER_SUNKEN), picture_panel(picture_panel),
        m_parent(parent), m_min_d(new wxSlider(this, ID_MIN_D, 300, 0, 10000, wxPoint(60, 10), wxSize(980, 15))),
        m_max_d(new wxSlider(this, ID_MAX_D, 1000, 0, 10000, wxPoint(60, 40), wxSize(980, 15))),
        m_min_d_text(new wxTextCtrl(this, ID_MIN_D_TEXT, "300", wxPoint(10, 10), wxSize(40, 15))),
        m_max_d_text(new wxTextCtrl(this, ID_MAX_D_TEXT, "1000", wxPoint(10, 40), wxSize(40, 15))) {
   Connect(ID_MIN_D, wxEVT_SCROLL_CHANGED, wxCommandEventHandler(SettingsPanel::on_change));
   Connect(ID_MAX_D, wxEVT_SCROLL_CHANGED, wxCommandEventHandler(SettingsPanel::on_change));
}

void SettingsPanel::on_change(wxCommandEvent &event) {
   int64_t const min_depth = m_min_d->GetValue();
   int64_t const max_depth = m_max_d->GetValue();

   m_min_d_text->Clear();
   m_min_d_text->WriteText(std::to_string(m_min_d->GetValue()));
   m_max_d_text->Clear();
   m_max_d_text->WriteText(std::to_string(m_max_d->GetValue()));
   // TODO: Make wxTextCtrls non-editable or update values on edit.

   // picture_panel->update_picture(min_depth, max_depth);
}

uint8_t *color_data = nullptr;

DisplayPanel::DisplayPanel(wxPanel *parent, Picture *picture)
      : wxPanel(parent, wxID_ANY, wxPoint(20, 20), wxSize(640, 480), wxBORDER_SUNKEN), picture(picture) {
   color_data = new uint8_t[640 * 480 * 3];
   Connect(wxID_ANY, REFRESH_PICTURE_EVENT, wxCommandEventHandler(DisplayPanel::refresh_picture));
}

void DisplayPanel::refresh_picture(wxCommandEvent &event) {
   delete m_picture;
   m_picture = new wxStaticBitmap(this, wxID_ANY, wxBitmap(wxImage(640, 480, color_data, true)), wxDefaultPosition, wxDefaultSize);
}

MainWindow::MainWindow(const wxString &title)
      : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(1900, 900)),
        picture(new Picture(nullptr, nullptr, nullptr)),
        m_parent(new wxPanel(this, wxID_ANY)), m_display(new DisplayPanel(m_parent, picture)),
        m_settings(new SettingsPanel(m_parent, m_display)) {

   auto *vbox = new wxBoxSizer(wxVERTICAL);
   vbox->Add(m_display, 1, wxEXPAND | wxALL, 5);
   vbox->Add(m_settings, 1, wxEXPAND | wxALL, 5);

   m_parent->SetSizer(vbox);
   Centre();
}

// Kinect handling

class MyKinectDevice : public KinectDevice {
 public:
   explicit MyKinectDevice(int device_number) : KinectDevice(device_number) {}

   void frame_handler(Picture const &picture) const override {
      Picture picture_copy(picture);
      picture_copy.resize_all(640, 480);
//      if (picture.color_frame) {
//         std::cout << "frame\n";
//         auto picture_copy = new Picture(picture);
//         picture_copy->resize_all(480, 360);
//
//         if (picture_copy->color_frame != nullptr) {
//            for (size_t i = 0; i < 480 * 360; ++i) {
//               color_data[3 * i]     = picture_copy->color_frame->pixels->data()[i].red;
//               color_data[3 * i + 1] = picture_copy->color_frame->pixels->data()[i].green;
//               color_data[3 * i + 2] = picture_copy->color_frame->pixels->data()[i].blue;
//            }
//         }
//
//         wxPostEvent(window->m_display, wxCommandEvent(REFRESH_PICTURE_EVENT));
//      }
      if (picture_copy.ir_frame) {
         float max_value = 0;
         for (size_t i = 0; i < 640 * 480; ++i) {
            max_value = std::max(max_value, picture_copy.ir_frame->pixels->data()[i]);
         }
         for (size_t i = 0; i < 640 * 480; ++i) {
            auto pixel_value = uint8_t(255.0 * picture_copy.ir_frame->pixels->data()[i] / max_value);
            color_data[3 * i] = pixel_value;
            color_data[3 * i + 1] = pixel_value;
            color_data[3 * i + 2] = pixel_value;
         }
         wxPostEvent(window->m_display, wxCommandEvent(REFRESH_PICTURE_EVENT));
      }
   }

   MainWindow *window = nullptr;
};

// Main

class AppMain : public wxApp {
 public:
   bool OnInit() override;
};

IMPLEMENT_APP(AppMain)

bool AppMain::OnInit() {
   MainWindow *window = new MainWindow(wxT("Simple display"));
   window->Show(true);
   auto kinect_device = new MyKinectDevice(0);
   kinect_device->window = window;
   kinect_device->start_streams(false, false, true);

   return true;
}
